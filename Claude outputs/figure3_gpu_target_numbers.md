# Figure 3 (routing) — mục tiêu số liệu khi chạy lại trên GPU RTX 4060 Ti

## 1. Vì sao mình đưa ra các con số này

Mình đã đọc trực tiếp `_routed_occupancy_solve` trong `rocu_net/model.py` (dòng 374 trở đi) để biết routing có thực sự giảm compute hay chỉ là mask hiển thị:

```python
selected_parent = parent_flat[active_flat]   # boolean indexing THẬT — tensor nhỏ hẳn lại
selected_scores = scores_flat[active_flat]
for _ in range(iterations):                   # 24 vòng, chạy trên tensor đã thu nhỏ
    ...
children_flat[active_flat] = torch.sigmoid(...)  # scatter ngược lại
```

Đây là **indexing thật** (`tensor[bool_mask]`), không phải `torch.where` che số — nên khi skip 98% cell, vòng lặp bisection thực sự chỉ chạy trên ~1-2% số cell đó. Điều này giải thích vì sao Dice không đổi và vì sao có một khoản latency thật được tiết kiệm (không phải noise).

Nhưng đồng thời, ngay ở chế độ **dense** (không routing), tổng số cell cần giải mỗi ảnh (80×80 + 160×160 ≈ 32.000 cell, batch=1) đã rất nhỏ so với phần compute chính là encoder/decoder conv dày đặc. Solver chưa bao giờ là bottleneck lớn — nên dù giảm 98% việc của nó, tổng latency toàn mô hình chỉ nhích nhẹ. Đây là lý do CPU chỉ thấy ~3%, và đây **không phải bug**, mà là hệ quả tự nhiên của kiến trúc.

Có thêm một chi tiết kỹ thuật: dòng `if bool(active_flat.any()):` bắt buộc đồng bộ GPU↔CPU (device sync) mỗi lần forward. Việc này xảy ra như nhau ở cả hai mode nên không làm lệch kết quả so sánh, nhưng có thể khiến số đo GPU nhiễu hơn nếu script không gọi `torch.cuda.synchronize()` đúng chỗ ngay trước/sau vòng đo — nên kiểm tra lại `scripts/figure3_routing.py` có làm đúng việc này không trước khi tin số liệu.

## 2. Dự đoán hợp lý cho từng chỉ số trên GPU

| Chỉ số | Số CPU đã có | Dự đoán hợp lý trên GPU | Diễn giải |
|---|---|---|---|
| Dice (mọi mode) | 0.88875, lệch nhau ~1e-7 | **Giữ nguyên gần như tuyệt đối** (lệch <1e-4, lý tưởng ~1e-6–1e-7) | Đây là phép toán xác định (bisection), không phụ thuộc phần cứng — nếu GPU cho Dice lệch nhiều hơn CPU, có bug (ví dụ FP16/TF32 rounding), cần kiểm tra lại `--device cuda` có bật đúng FP32 không. |
| Solver skip fraction | 98–99% | **Gần như y hệt** (97–99%) | Đây là hàm của *mô hình đã train* + ngưỡng τᵣ, không phụ thuộc CPU/GPU. Nếu số này lệch nhiều so với CPU, có nghĩa checkpoint hoặc input pipeline không khớp. |
| Latency giảm so với Dense | ~2–3% (τᵣ=0.05–0.20), ~1.9% (τᵣ=0.30) | **Khả năng cao vẫn khiêm tốn: 2–8%** | Vì kernel-launch overhead trên GPU gần như cố định bất kể tensor to hay nhỏ (mỗi lần launch ~5–20µs), lợi ích từ việc "tensor nhỏ hơn" bị pha loãng. Đừng kỳ vọng con số kiểu 30-50% trừ khi có tối ưu thêm (xem mục 4). |
| FPS | ~9.6–9.9 | Cao hơn nhiều lần (GPU nhanh hơn CPU tổng thể), nhưng **tỉ lệ % cải thiện giữa dense vs routed** nhiều khả năng vẫn nhỏ tương tự | Đừng nhầm "FPS cao" (do GPU nhanh) với "routing hiệu quả" (do % giảm so với dense) — hai chuyện khác nhau, paper cần chỉ số thứ hai. |

## 3. Số liệu nào mới thực sự là "support mạnh"

**Đừng đặt mục tiêu là ép latency giảm thật nhiều** — nếu số thật ra chỉ 3-5% thì cứ báo cáo đúng vậy, đó vẫn là kết quả hợp lệ và trung thực (giống báo cáo CPU đã viết sẵn). Thứ thực sự "support mạnh" cho bài báo, theo thứ tự ưu tiên:

1. **Dice không đổi khi routing** (đã có, cực mạnh) — chứng minh cơ chế confidence-routing an toàn về mặt số học, đúng như constraint đề ra. Đây là claim chính, không phải latency.
2. **Solver skip fraction cao và ổn định qua nhiều τᵣ** (đã có, ~98%) — chứng minh phần lớn cell là "tự tin", cơ chế route đúng chỗ.
3. **Latency giảm — dù nhỏ — theo đúng chiều kỳ vọng và nhất quán qua 5 lần lặp** (đã có trên CPU) — đủ để nói "có lợi ích thực, dù bị giới hạn bởi phần dense phía sau", không cần con số ấn tượng.
4. **(Tuỳ chọn, làm bài mạnh hơn nhiều nếu có thời gian): báo cáo thêm số liệu FLOPs/MACs mà solver tiết kiệm được**, thay vì chỉ latency. Vì đây là chỉ số xác định (không phụ thuộc kernel-launch overhead của phần cứng), một con số như "giảm ~98% MACs của riêng khối occupancy solver" sẽ mạnh và "sạch" hơn nhiều so với con số latency vốn dễ bị nhiễu bởi phần cứng. Nếu `profiler.py` trong repo hỗ trợ đếm MACs riêng cho solver (có/không cần kiểm tra), đây là số nên thêm song song với latency.

Nói cách khác: **latency là bằng chứng phụ, Dice-preservation + skip-fraction (+ lý tưởng là FLOPs solver) mới là bằng chứng chính.** Nếu GPU cho latency giảm ít hơn cả CPU, vẫn có thể publish được — chỉ cần diễn giải đúng như báo cáo CPU đã làm rất tốt rồi (đừng xoá đoạn giải thích "modest vì phần dense vẫn chiếm phần lớn").

## 3b. Đã tìm thêm 2 dữ kiện quan trọng từ `runs/rocu_kvasir_seed42/lightweight_metrics.json`

Đây là file profiler THẬT đã chạy trên chính RTX 4060 Ti của bạn (không phải CPU), nên mình dùng nó làm điểm neo (anchor) cho bảng mục tiêu bên dưới:

- **GMACs/GFLOPs khớp chính xác với §4.5**: `gmacs_per_image = 1.749475584`, `gflops_per_image = 3.498951168` — đúng bằng 1.7495 GMACs / 3.4990 GFLOPs mà prose Experiments đang ghi. Vậy **prose §4.5 là ĐÚNG, số cần sửa là Table 2 (đang ghi "1.1G")** — không phải ngược lại như mình từng để ngỏ trong `% AUTHOR CHECK`. Đây coi như giải quyết xong khoản "chưa đối chiếu được FLOPs/GMACs" mình từng nêu.
- **Cảnh báo quan trọng**: cùng file này có `"confidence_routing": {"stage1_active_fraction": 0.0, "stage2_active_fraction": 0.0, "solver_skip_fraction_estimate": 1.0}` — tức là lần profile latency GPU này chạy trên **input ngẫu nhiên (synthetic)** và routing tình cờ skip 100% solver (không phải hành vi thật trên ảnh nội soi thật, nơi skip chỉ ~98%). Vì vậy con số latency 10.398 ms/ảnh (96.17 FPS) trong file này **không phải "dense" hợp lệ để so sánh** — nó gần với biên dưới lý tưởng (gần như toàn bộ solver bị bỏ qua). Tuyệt đối không dùng số này làm "Dense" trong Figure 3 — chỉ dùng làm điểm neo tham khảo cho bảng mục tiêu bên dưới.

## 3c. Bảng mục tiêu cụ thể cho lần chạy GPU thật (real-image, `scripts/figure3_routing.py --device cuda`)

| Mode | Dice — bắt buộc giữ | Solver skipped — kỳ vọng | Latency dự đoán (ms/ảnh) | FPS dự đoán | Δ latency vs Dense (dự đoán) |
|---|---|---|---|---|---|
| Dense | = mốc so sánh | 0% (định nghĩa) | ~10.6 – 13.5 | ~74 – 94 | — |
| τᵣ=0.05 | lệch <1e-4, lý tưởng ~1e-6–1e-7 (giống CPU) | 96 – 99% (CPU: 97.99%) | ~10.4 – 11.8 | ~85 – 96 | −3% đến −15% |
| τᵣ=0.10 *(ngưỡng đang dùng trong `configs/kvasir.yaml`)* | lệch <1e-4 | 96 – 99% (CPU: 98.38%) | ~10.4 – 11.8 | ~85 – 96 | −3% đến −15% |
| τᵣ=0.20 | lệch <1e-4 | 97 – 99.5% (CPU: 98.80%) | ~10.4 – 11.9 | ~84 – 96 | −2% đến −14% |
| τᵣ=0.30 | lệch <1e-4, SD giữa 5 lần lặp có thể cao hơn (giống CPU ±3.13ms) | 98 – 99.7% (CPU: 99.18%) | ~10.5 – 12.5 | ~80 – 95 | −2% đến −12% |

Cách đọc bảng này:
- Cột Dice và Solver skipped là **ràng buộc phải đạt** — nếu lệch nhiều khỏi khoảng này, khả năng cao là bug (sai checkpoint, sai FP32/TF32, sai split) chứ không phải do GPU khác CPU.
- Cột Latency/FPS là **dự đoán có lý do** (suy từ điểm neo 10.4ms ở trên + đặc tính kernel-launch-overhead của GPU đã giải thích ở mục 1), **không phải con số cần đạt bằng mọi giá**. Số thật có thể nằm ngoài khoảng này — cứ báo cáo đúng số đo được.
- Nếu số đo thật cho Δ latency nằm ở đầu dưới khoảng dự đoán (gần −15%) → đây là kịch bản "mạnh", nên dùng làm số chính trong bài. Nếu chỉ đạt −2% đến −5% → vẫn hợp lệ, diễn giải giống hệt cách CPU report đã viết (lợi ích thật nhưng khiêm tốn vì phần dense phía sau chiếm phần lớn compute).
- Nếu Δ latency ra **dương** (routed chậm hơn dense) ở một vài mode — cũng là kết quả có thể xảy ra và vẫn báo cáo trung thực được, kèm giải thích overhead của boolean-indexing/gather trên GPU với tensor cực nhỏ đôi khi không bù được compute tiết kiệm.

## 4. Checklist khi chạy trên server

- Dùng đúng lệnh có sẵn trong `docs/FIGURE3_PAPER_REPORT.md`:
  ```bash
  python scripts/figure3_routing.py \
    --checkpoint runs/rocu_kvasir_seed42/best.pt \
    --data-root dataset/Kvasir-SEG \
    --split val --device cuda --cpu-threads 6 \
    --illustration-threshold 0.10 \
    --warmup 30 --repeats 5 --batch-size 1 --seed 42 --dpi 600 \
    --output figures/kvasir_routing_paper_gpu
  ```
- Đảm bảo GPU rảnh hoàn toàn khi đo (không chạy job khác song song) — nhiễu latency GPU thường đến từ tranh chấp tài nguyên chứ không phải từ routing.
- Nếu độ lệch chuẩn (SD) giữa 5 lần lặp lớn (như τᵣ=0.30 ở bản CPU: ±3.13ms, khá cao so với chênh lệch mean chỉ ~1-3ms), cân nhắc tăng `--repeats` lên 10-15 để CI chặt hơn — hiện tại chênh lệch latency giữa các mode (~1-3ms) khá gần với SD, nên kết luận "routing nhanh hơn" cần nhiều lần lặp hơn mới vững.
- Giữ nguyên `--seed 42` và cùng checkpoint/split để so sánh táo với táo với bản CPU.
- Báo cáo đúng số đo được — không chọn τᵣ hay lần lặp có lợi nhất để "làm đẹp" bảng.
