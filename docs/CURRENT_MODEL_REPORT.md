# HiGAN+ Current Model Report

## 1. Mục tiêu của report

Tài liệu này mô tả trạng thái hiện tại của model trong workspace `higanplus/HiGANplus` trước khi bước sang giai đoạn nâng cấp và research cải tiến.

Phạm vi của report gồm:

- Kiến trúc hiện tại của hệ thống HiGAN+.
- Các thành phần phụ trợ đang tham gia train và inference.
- Pipeline dữ liệu, checkpoint, config, metrics đánh giá.
- Khác biệt giữa upstream paper/code và fork hiện tại.
- Hành vi runtime đã được quan sát khi chạy local và Kaggle.
- Các giới hạn, bug, điểm lệch cấu hình, và các rủi ro kỹ thuật cần biết trước khi nghiên cứu tiếp.

Report này cố ý tập trung vào trạng thái thực tế của code hiện có, không chỉ mô tả paper ở mức khái niệm.

## 2. Executive Summary

HiGAN+ hiện tại là một hệ thống sinh ảnh chữ viết tay có điều kiện theo nội dung văn bản và phong cách viết, với kiến trúc chính gồm:

- `Generator` sinh ảnh chữ viết tay từ `style vector + text labels`.
- `StyleEncoder` trích style latent từ ảnh tham chiếu.
- `StyleBackbone` làm backbone đặc trưng dùng chung cho style encoder và writer identifier.
- `Discriminator` toàn cục và `PatchDiscriminator` cục bộ.
- `Recognizer` OCR để cung cấp CTC loss cho khả năng đọc được.
- `WriterIdentifier` để ép phong cách người viết.

Về bản chất, model không chỉ là một GAN đơn thuần. Nó là một hệ gồm 1 generator, 2 discriminator, 1 style encoder kiểu VAE, 1 OCR phụ trợ, 1 writer classifier phụ trợ, và một tập losses cân bằng bằng gradient statistics.

Điểm quan trọng nhất cần hiểu trước khi research tiếp:

- `gan_iam.yml` là chế độ train GAN chính từ random init cho `G/D/E`, nhưng vẫn dùng `Recognizer` và `WriterIdentifier` pretrained làm auxiliary teachers.
- `gan_iam_kaggle.yml` là chế độ fine-tune từ `deploy_HiGAN+.pth`, không phải train từ đầu cho generator.
- Checkpoint `deploy_HiGAN+.pth` chỉ chứa `Generator + StyleEncoder + StyleBackbone`, không chứa discriminator, recognizer, writer identifier, hay optimizer states.
- Pipeline train full “từ đầu toàn bộ hệ” hiện chưa khép kín trong fork vì config OCR/WID yêu cầu dataset `iam_word` (`trnvalset_words64.hdf5`), trong khi script setup hiện chỉ tải `iam_word_org` (`trnvalset_words64_OrgSz.hdf5`).

## 3. Trạng thái fork hiện tại

Fork hiện tại không còn là code upstream nguyên bản. Nó đã được chỉnh để chạy trên stack hiện đại hơn và để phục vụ local/Kaggle workflow.

Các thay đổi đáng chú ý trong fork:

- Compat với PyTorch 2.x trong `networks/rand_dist.py` bằng cách bổ sung `new_empty()` và `__deepcopy__()` cho subclass `Distribution`.
- Compat với torchvision mới trong `lib/utils.py`: `make_grid(range=...)` đã đổi sang `value_range=...`.
- `train.py` tự set `PYTORCH_ALLOC_CONF=expandable_segments:True` và alias cũ `PYTORCH_CUDA_ALLOC_CONF` trước khi import torch để giảm fragmentation trên CUDA.
- `run_demo.py` là runner headless, không cần `input()` hoặc `plt.show()`.
- `scripts/setup_data.sh` tải data IAM và 4 checkpoint upstream.
- Bổ sung các config smoke/Kaggle/scratch để test nhanh, fine-tune, hoặc kiểm tra from-scratch behavior.
- `GlobalLocalAdversarialModel`, `RecognizeModel`, và `WriterIdentifyModel` đều đã hỗ trợ `max_iters_per_epoch` cho smoke tests.

Các commit gần đây phản ánh hướng phát triển này:

- `3e8aad7` Compat cho PyTorch 2.x + Pillow 10+.
- `2e5804f` Thêm tooling Kaggle/setup/demo headless.
- `83088c3` Warm-start phụ trợ W/B/R ổn định hơn.
- `18dd868` Fix torchvision `value_range`, smoke config, và giới hạn iter/epoch.
- `f52778a`, `7fca0de`, `77b5aad`, `67472f0` thêm notebook/config cho Kaggle, giảm batch để tránh OOM, và thêm config scratch test.

## 4. Inventory hiện có trong repo

### 4.1. Dữ liệu hiện có

Thư mục `HiGAN+/data/iam/` hiện có:

- `trnvalset_words64_OrgSz.hdf5` khoảng 270 MB.
- `testset_words64_OrgSz.hdf5` khoảng 56 MB.

Ý nghĩa:

- Đây là dataset word-level, ảnh chiều cao 64 px, giữ chiều rộng gốc, dùng cho `iam_word_org`.
- Đây là dataset mà GAN chính (`gan_iam.yml`, `gan_iam_kaggle.yml`, `gan_iam_scratch.yml`) đang dùng.

### 4.2. Checkpoint hiện có

Thư mục `HiGAN+/pretrained/` hiện có:

- `deploy_HiGAN+.pth` khoảng 22 MB.
- `ocr_iam_new.pth` khoảng 9.4 MB.
- `wid_iam_new.pth` khoảng 6.9 MB.
- `wid_iam_test.pth` khoảng 6.7 MB.

Vai trò của từng file:

- `deploy_HiGAN+.pth`: bundle deploy rút gọn cho suy luận/fine-tune generator-side.
- `ocr_iam_new.pth`: recognizer OCR pretrained.
- `wid_iam_new.pth`: writer identifier pretrained cho tập train/validation chính.
- `wid_iam_test.pth`: writer identifier test-time dùng cho metric WIER trên split test.

### 4.3. Cách tạo deploy checkpoint

File `deploy_checkpoint.py` xác nhận rất rõ nội dung của deploy checkpoint.

`deploy_HiGAN+.pth` chỉ giữ lại 3 phần:

- `Generator`
- `StyleEncoder`
- `StyleBackbone`

Nghĩa là khi fine-tune từ `deploy_HiGAN+.pth`, việc thấy log dạng sau là bình thường:

- `Load Discriminator failed`
- `Load PatchDiscriminator failed`
- `Load Recognizer failed`
- `Load WriterIdentifier failed`
- `Load OPT.G failed`
- `Load OPT.D failed`

Đây không phải crash logic. Đây là hậu quả trực tiếp của việc file deploy không chứa các state đó.

## 5. Môi trường và dependencies

### 5.1. Dependency Python

`requirements.txt` hiện gồm:

- `numpy<2.0`
- `Pillow`
- `matplotlib`
- `tqdm`
- `PyYAML`
- `munch`
- `h5py`
- `tensorboard`
- `opencv-python`
- `scikit-image`
- `scipy`
- `distance`
- `einops`

Torch không nằm trong `requirements.txt` vì local/Kaggle có thể cài riêng theo CUDA build.

### 5.2. Runtime assumptions

Code hiện kỳ vọng:

- GPU CUDA sẵn sàng.
- Dataloader chạy với `num_workers=4` ở nhiều chỗ.
- Ảnh grayscale 1 channel, background ở mức `-1` sau normalize.
- Tập ký tự `n_class=80` theo `lib/alphabet.py`.

### 5.3. Memory handling

`train.py` hiện tự set:

- `PYTORCH_ALLOC_CONF=expandable_segments:True`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

Lý do là G-step của HiGAN+ tạo peak memory rất dao động theo độ dài từ, dễ gây fragmentation trên T4 15 GB.

## 6. Kiến trúc tổng thể của hệ thống

### 6.1. Ba model top-level trong repo

File `networks/__init__.py` expose 3 model runner:

- `gl_adversarial_model` -> `GlobalLocalAdversarialModel`
- `recognize_model` -> `RecognizeModel`
- `identifier_model` -> `WriterIdentifyModel`

Nói cách khác, repo hiện chứa 3 stage logic:

1. OCR recognizer training.
2. Writer identifier training.
3. HiGAN+ adversarial training.

### 6.2. Thành phần của `GlobalLocalAdversarialModel`

Trong `networks/model.py`, runner GAN chính instantiate:

- `G`: `Generator`
- `D`: `Discriminator`
- `P`: `PatchDiscriminator`
- `R`: `Recognizer`
- `E`: `StyleEncoder`
- `W`: `WriterIdentifier`
- `B`: `StyleBackbone`

Tập losses gắn với runner này:

- `CTCLoss`
- `CrossEntropyLoss`
- `CXLoss` (contextual loss)

## 7. Generator hiện tại

### 7.1. Kiểu generator

Generator lấy cảm hứng từ BigGAN nhưng đã chỉnh cho bài toán handwriting synthesis.

Input chính:

- `z`: style vector, mặc định `style_dim=32`.
- `y`: sequence label đã encode theo từng ký tự.
- `y_lens`: độ dài sequence.

### 7.2. Cơ chế gắn text vào generator

Ở `Generator.forward()`:

- Ký tự được nhúng qua `nn.Embedding(n_class=80, embed_dim=120)`.
- Style vector `z` được repeat theo chiều dài chuỗi ký tự.
- Text embedding và style vector được concat tại từng vị trí ký tự.
- Kết quả được chiếu qua `filter_linear` để tạo tensor khởi tạo dạng “vertical patches per character”.

Kiến trúc này rất quan trọng vì nó giải thích vì sao output width phụ thuộc trực tiếp vào số ký tự.

### 7.3. Hình học output

Với config mặc định:

- `bottom_width = 4`
- Chuỗi ký tự dài `L`
- Các upsample theo chiều rộng là `(1, 2, 2, 2)` trên 4 block

Suy ra output width cuối cùng xấp xỉ:

- `L * 4 * 1 * 2 * 2 * 2 = 32 * L`

Điều này khớp với `char_width = 32` trong config.

Chiều cao:

- `bottom_height = 4`
- upsample chiều cao theo `2 * 2 * 2 * 2 = 16`
- output height = `4 * 16 = 64`

Đây là lý do mọi ảnh sinh ra có chiều cao cố định 64 px.

### 7.4. Đặc điểm thiết kế

- Spectral normalization trên conv/linear khi `G_param='SN'`.
- Conditional batch norm kiểu `ccbn` theo style chunks.
- Optional self-attention, nhưng config hiện tại `G_attn='0'`, nghĩa là không bật attention layer nào trong generator.

## 8. Style encoder và style backbone

### 8.1. `StyleBackbone`

`StyleBackbone` là CNN backbone dùng chung cho:

- `StyleEncoder`
- `WriterIdentifier`

Nó giảm chiều dài feature theo scale `16` (`reduce_len_scale = 16`).

Backbone trả về:

- `out`: đặc trưng sequence sau `cnn_ctc`
- `feats`: các feature maps trung gian nếu `ret_feats=True`

Các feature maps trung gian này được dùng cho contextual loss.

### 8.2. `StyleEncoder`

`StyleEncoder`:

- Lấy đặc trưng từ `StyleBackbone`.
- Mask theo chiều dài ảnh.
- Average pooling theo trục width để ra 1 vector style cho mỗi ảnh.
- Qua MLP 2 tầng để sinh `mu` và `logvar`.

Nếu `vae_mode=True`:

- `style = reparameterize(mu, logvar)`
- training có thêm `KLloss`

Nếu `vae_mode=False`:

- style dùng trực tiếp `mu`

Config hiện tại luôn bật `vae_mode: true` trong các config GAN chính.

## 9. Discriminator toàn cục và patch discriminator

### 9.1. `Discriminator`

Discriminator chính là discriminator toàn ảnh theo độ dài word.

Đặc điểm:

- BigGAN-style residual discriminator.
- Dùng spectral norm.
- Không dùng class projection theo nội dung ký tự; thay vào đó điều kiện text đi vào generator, còn discriminator đánh giá ảnh theo vùng hợp lệ và độ dài.
- Nếu có `x_lens`, discriminator mask phần không hợp lệ theo chiều dài thực của ảnh.

### 9.2. `PatchDiscriminator`

`PatchDiscriminator` kế thừa trực tiếp `Discriminator`, nhưng làm việc trên patches `32x32` đã được cắt sẵn từ ảnh.

Patch extraction nằm ở `extract_all_patches()` trong `networks/utils.py`:

- `block_size = 32`
- `step = 8`
- Số patches tăng theo chiều dài ảnh

Đây là một nguyên nhân chính gây biến động memory theo độ dài word.

### 9.3. Vai trò của hai discriminator

- `D` ép chất lượng toàn cục ở cấp độ ảnh chữ.
- `P` ép texture/cấu trúc cục bộ ở mức patch.

Loss discriminator tổng:

- `real_disc_loss`
- `fake_disc_loss`
- `real_disc_loss_patch`
- `fake_disc_loss_patch`

Tất cả được cộng trực tiếp theo hinge loss.

## 10. Recognizer OCR và Writer Identifier

### 10.1. Recognizer

`Recognizer` là CNN + optional BLSTM + linear classifier cho CTC.

Cấu hình hiện tại:

- `len_scale = 16`
- `rnn_depth = 2`
- `bidirectional = True`
- `n_class = 80`

Recognizer cung cấp:

- `CTC loss` trên ảnh fake
- metric `CER/WER` khi validate/test

### 10.2. WriterIdentifier

`WriterIdentifier` nhận feature từ `StyleBackbone`, average pooling theo width, rồi qua MLP classification để dự đoán writer ID.

Nó cung cấp:

- `CrossEntropy` loss để ép style consistency
- metric `WIER` khi đánh giá generator

Lưu ý về `WIER` trong code hiện tại:

- Đây không phải supervised writer-ID error trực tiếp so với nhãn ground truth của ảnh fake.
- `validate_wid()` dự đoán writer label trên ảnh thật, dự đoán lại trên ảnh fake, rồi đo mức bất đồng giữa hai dự đoán đó.
- Vì vậy, `WIER` nên được hiểu là một proxy metric cho style-consistency, không phải writer classification error thuần túy.

### 10.3. Vai trò thật sự của R, W và B

Đây là điểm dễ gây hiểu nhầm nhất.

Trong train GAN chính:

- `R` và `W` không phải phần sinh ảnh chính.
- `B` cũng không khởi tạo độc lập trong workflow chuẩn, vì thường được warm-start từ checkpoint writer hoặc deploy checkpoint.
- Ba module `R/W/B` cùng hoạt động như teacher / auxiliary support để đưa signal nội dung và phong cách.

Vì vậy:

- Train GAN chính với `pretrained_r` + `pretrained_w` vẫn được xem là quy trình chuẩn của repo/paper, đồng thời checkpoint writer cũng seed luôn `StyleBackbone`.
- Nếu muốn “full scratch tuyệt đối”, phải train cả `R`, `W`, và backbone phụ trợ tương ứng từ đầu trước.

## 11. Pipeline dữ liệu hiện tại

### 11.1. Dataset format

`Hdf5Dataset` load các trường chính từ file `.hdf5`:

- `imgs`
- `lbs`
- `img_seek_idxs`
- `lb_seek_idxs`
- `img_lens`
- `lb_lens`
- `wids`

Mỗi sample bao gồm:

- ảnh gốc grayscale
- text string
- label sequence
- writer ID

### 11.2. Hai cách biểu diễn ảnh trong batch

Một sample có thể sinh ra 3 loại ảnh:

- `org_img`: ảnh gốc giữ width thật
- `style_img`: ảnh resize về width = `CharWidth * len(text)` nếu `process_style=True`
- `aug_img`: ảnh qua augmentation nếu train OCR hoặc WID

`style_img` là chìa khóa để nối style feature với text length chuẩn hóa theo số ký tự.

### 11.3. Padding rule

`collect_fn()` pad width về bội số của `CharWidth = 32`.

Điều này giúp:

- Generator/recognizer/discriminator có chiều dài tương thích nhau.
- Masking theo độ dài ảnh ổn định hơn.

### 11.4. Augmentation

Augmentations chính nằm ở `lib/transforms.py`:

- `RandomScale`: scale width ngẫu nhiên
- `RandomClip`: crop ngẫu nhiên theo width nếu ảnh đủ dài

Quy ước dùng augmentation:

- OCR training: thường dùng `RandomScale`
- WID training: thường dùng `RandomClip`
- GAN training: dataset generator dùng cả `recogn_aug=True`, `wid_aug=True`, `process_style=True`

## 12. Losses của GAN chính

### 12.1. Thành phần loss

Trong G-step, repo hiện dùng các thành phần sau:

- `adv_loss`: global adversarial loss từ `D`
- `adv_loss_patch`: patch adversarial loss từ `P`
- `fake_ctc_loss`: OCR/CTC loss trên random fake, style-guided fake, reconstruction fake
- `info_loss`: ép style encode lại ảnh fake gần với `z` đã sample
- `recn_loss`: L1 reconstruction giữa `recn_imgs` và `real_imgs`
- `fake_wid_loss`: writer classification loss trên `style_imgs` và `recn_imgs`
- `ctx_loss`: contextual loss giữa feature thật và feature fake
- `kl_loss`: KL divergence của VAE style encoder

### 12.2. Cân bằng loss bằng gradient statistics

Một nét đặc biệt của code hiện tại là không dùng các weight scalar cố định cho mọi auxiliary loss. Thay vào đó, code ước lượng độ lệch chuẩn gradient và tạo các hệ số động:

- `gp_ctc`
- `gp_info`
- `gp_wid`
- `gp_recn`

Các hệ số này được tính bằng tỉ lệ giữa `std_grad_adv` và `std` của gradient từ từng auxiliary loss hoặc branch đại diện của auxiliary loss đó.

Generator loss cuối cùng:

`g_loss = adv + adv_patch + gp_ctc * ctc + gp_info * info + gp_wid * wid + gp_recn * recn + lambda_ctx * ctx + lambda_kl * kl`

Chi tiết cần lưu ý:

- `gp_ctc` hiện được ước lượng từ branch `fake_ctc_loss_rand`, rồi mới áp vào tổng `fake_ctc_loss` gồm random + style + reconstruction.
- Nếu sau này muốn sửa cơ chế loss balancing, đây là một điểm rất đáng kiểm tra lại.

Đây là một đặc điểm quan trọng của baseline hiện tại. Nếu nghiên cứu cải tiến, đây là một trong những nơi đáng kiểm chứng nhất.

### 12.3. Contextual loss

`CXLoss` dùng patch-based cosine similarity giữa feature maps thật và fake. Nó khá nặng về memory vì phải xây raw distance tensor và tương đối hóa trên patches.

Đây là một trong các nguyên nhân chính gây áp lực memory trên Kaggle T4 khi batch lớn.

### 12.4. Một lệch cấu hình đáng chú ý

Config có `lambda_gram`, và file `networks/loss.py` có `GramStyleLoss`, nhưng trong `GlobalLocalAdversarialModel.train()` hiện tại không có chỗ nào cộng gram loss vào `g_loss`.

Nói cách khác:

- `lambda_gram` hiện tồn tại trong config nhưng không được dùng thật.
- `GramStyleLoss` hiện là code chết theo nghĩa runtime ở đường train GAN chính.

Đây là một discrepancy rất đáng ghi nhận trước khi nghiên cứu cải tiến.

## 13. Training modes hiện có

### 13.1. `configs/gan_iam.yml`

Đây là config chuẩn để train GAN chính trên `iam_word_org`.

Thông số nổi bật:

- `epochs = 70`
- `batch_size = 8`
- `pretrained_ckpt = ''`
- `pretrained_w` và `pretrained_r` vẫn được load

Ý nghĩa:

- `G/D/E` bắt đầu random
- `R/W/B` được warm-start bằng checkpoint phụ trợ

Đây là “from scratch cho GAN chính”, không phải “full pipeline scratch”.

### 13.2. `configs/gan_iam_finetune.yml`

Config fine-tune ngắn hơn:

- `epochs = 30`
- `batch_size = 4`
- `lr = 1e-4`
- `pretrained_ckpt = './pretrained/deploy_HiGAN+.pth'`

Phù hợp khi muốn bắt đầu từ model deploy đã đẹp sẵn.

### 13.3. `configs/gan_iam_kaggle.yml`

Config Kaggle hiện tại đã được tune để tránh OOM trên T4:

- `epochs = 70`
- `batch_size = 4`
- `sample_nrow = 2`
- `save_epoch_val = 5`
- `pretrained_ckpt = './pretrained/deploy_HiGAN+.pth'`

Đây là config fine-tune / production-ish cho Kaggle GPU nhỏ.

### 13.4. `configs/gan_iam_smoke.yml`

Config smoke test:

- `epochs = 3` nhưng thực chạy 2 epoch do `range(1, epochs)`
- `batch_size = 4`
- `max_iters_per_epoch = 30`
- `pretrained_ckpt = ''`
- `validate_before_train = false`
- `start_save_epoch_val = 99`

Dùng để verify loop train, sample image, và save checkpoint nhanh.

### 13.5. `configs/gan_iam_scratch.yml`

Config scratch test trên Kaggle:

- `epochs = 5`
- `batch_size = 4`
- `pretrained_ckpt = ''`
- vẫn giữ `pretrained_r` và `pretrained_w`

Mục đích là quan sát behavior của generator khi không warm-start từ deploy checkpoint, chứ không nhằm đạt chất lượng cuối tốt.

## 14. OCR/WID training pipeline

### 14.1. `configs/ocr_iam.yml`

Recognizer config:

- `model = 'recognize_model'`
- `dataset = 'iam_word'`
- `epochs = 74`
- `batch_size = 128`
- `augment = True`

### 14.2. `configs/wid_iam.yml`

Writer identifier config:

- `model = 'identifier_model'`
- `dataset = 'iam_word'`
- `epochs = 74`
- `batch_size = 32`
- `pretrained_backbone = './pretrained/ocr_iam_new.pth'`

Điều này cho thấy thứ tự logic chuẩn là:

1. train OCR
2. dùng OCR backbone để train WID
3. dùng OCR + WID đã train để train HiGAN+

### 14.3. Vấn đề hiện tại với “full scratch”

Hiện trạng workspace có một block kỹ thuật quan trọng:

- `ocr_iam.yml` và `wid_iam.yml` dùng dataset `iam_word`
- `iam_word` map tới `trnvalset_words64.hdf5` và `testset_words64.hdf5`
- `scripts/setup_data.sh` chỉ tải `trnvalset_words64_OrgSz.hdf5` và `testset_words64_OrgSz.hdf5`

Hệ quả:

- GAN chính chạy được ngay với data hiện có.
- OCR/WID training chuẩn theo config upstream không chạy được ngay với data hiện có.
- Vì vậy, full pipeline scratch hiện chưa đóng kín bằng script setup mặc định.

Đây là điểm quan trọng nhất cần giải quyết nếu phase research sắp tới muốn train lại toàn bộ hệ từ đầu.

## 15. Inference và evaluation

### 15.1. `run_demo.py`

Đây là demo headless hiện tại.

Input:

- `--config`
- `--ckpt`
- `--text`
- `--nrow`
- `--out`
- `--device`

Nó load checkpoint, sample style noise ngẫu nhiên, gọi trực tiếp `model.models.G(...)`, rồi save PNG. Đây là entrypoint inference tiện nhất hiện nay.

### 15.2. `eval_demo.py`

Demo tương tác qua terminal, hỗ trợ 4 mode:

- `rand`
- `style`
- `interp`
- `text`

Phù hợp để khám phá model bằng tay, nhưng không tối ưu cho headless/Kaggle do dùng `input()` và `plt.show()`.

Caveat quan trọng của fork hiện tại:

- `eval_demo.py` mặc định trỏ tới `./pretrained/HiGAN+.pth`.
- `scripts/setup_data.sh` hiện chỉ tải `deploy_HiGAN+.pth`, không tải `HiGAN+.pth` đầy đủ.
- Vì vậy, nếu dùng `eval_demo.py` theo setup hiện tại, cần override `--ckpt ./pretrained/deploy_HiGAN+.pth` hoặc tự chuẩn bị full checkpoint phù hợp.

### 15.3. `test.py`

Evaluation script dùng `model.validate(..., test_stage=True)` để tính:

- `FID`
- `KID`
- `IS`
- `PSNR`
- `MSSIM`
- `CER`
- `WER`
- `WIER` nếu style-guided

Giống `eval_demo.py`, `test.py` cũng mặc định trỏ tới `./pretrained/HiGAN+.pth`, nên trong setup hiện tại thường phải truyền `--ckpt ./pretrained/deploy_HiGAN+.pth` hoặc một full checkpoint khác.

### 15.4. Metrics hiện tại

`validate()` dùng:

- Inception-based feature statistics cho `FID/KID/IS`
- Recognizer OCR cho `CER/WER`
- WriterIdentifier cho `WIER`
- `PSNR/MSSIM` khi test-stage và không dùng rand corpus

Nhìn từ góc độ research, đây là baseline metric stack khá đầy đủ, nhưng cũng khá nặng về compute.

## 16. Hành vi runtime đã quan sát

### 16.1. Smoke training local

Đã verify local:

- `gan_iam_smoke.yml` chạy thành công 2 epoch x 30 iter
- tạo sample image
- tạo `last.pth`
- log và tensorboard hoạt động

Điều này xác nhận core loop `G/D/E/R/W/B` của GAN chính đang chạy được trên local environment hiện tại.

### 16.2. Fine-tune trên Kaggle

Từ các lần chạy thực nghiệm đã quan sát trên Kaggle T4 15 GB:

- `batch_size = 16` đã từng OOM ở đường train generator khi memory pressure cao
- `batch_size = 8` vẫn có thể OOM muộn hơn do peak memory dao động theo độ dài word và fragmentation
- `batch_size = 4` là mức đã được tune để ổn định hơn

Nguyên nhân cốt lõi theo code path hiện tại:

- G-step tạo `cat_fake_imgs = [fake, style, recn]`, tức batch hiệu quả là `3 x batch_size`
- patch count tăng theo chiều dài ảnh
- contextual loss và patch discriminator đều có thể làm peak memory không đều theo iter

### 16.3. Dấu hiệu phân biệt fine-tune và scratch

Khi fine-tune từ `deploy_HiGAN+.pth`, ngay iter đầu tiên có thể thấy:

- `CTC-fake` rất thấp
- `Recn-c` thấp
- sample image đẹp rất sớm

Khi bỏ `pretrained_ckpt` và chạy scratch behavior:

- `CTC-fake` cao, cỡ ~25 ở iter đầu
- `Recn-c` cao hơn rõ rệt
- ảnh sinh ra chưa đẹp trong vài nghìn iter đầu

Điều này đã được quan sát trực tiếp khi test local.

## 17. Strengths của baseline hiện tại

- Kiến trúc đã đủ đầy cho bài toán handwriting synthesis có điều kiện style + content.
- Có cả global discriminator và patch discriminator.
- Có auxiliary supervision mạnh từ OCR và writer ID.
- Có VAE-style style encoder thay vì latent noise thuần.
- Có metric stack tương đối đầy đủ cho research baseline.
- Fork hiện tại đã usable trên local hiện đại và trên Kaggle.
- Có headless demo runner và setup script tương đối tiện.

## 18. Hạn chế và technical debt hiện tại

### 18.1. `lambda_gram` hiện không có hiệu lực runtime

Config có `lambda_gram`, code có `GramStyleLoss`, nhưng train loop hiện không dùng gram loss.

Đây là một lệch cấu hình thực sự.

### 18.2. Full pipeline scratch chưa khép kín

OCR/WID config yêu cầu `iam_word`, nhưng setup chỉ tải `iam_word_org`.

Điều này ngăn việc reproduce trọn pipeline scratch chỉ bằng setup hiện tại.

### 18.3. Memory footprint rất nhạy với độ dài word

HiGAN+ không chỉ nhạy với batch size mà còn nhạy với distribution độ dài từ trong batch.

Điều này làm cho:

- peak memory khó dự đoán
- Kaggle T4 dễ OOM nếu batch cao
- timing giữa các iter không ổn định

### 18.4. Validation khá đắt

`validate()` tính FID/KID/IS, có thể thêm CER/WER/WIER/PSNR/MSSIM. Đây là pipeline nặng.

Với Kaggle hoặc GPU nhỏ, validate mỗi epoch không thực tế.

### 18.5. `sample_images()` có một bug padding đáng chú ý

Trong `sample_images()`:

- `style_imgs` đang được pad bằng `max_img_len - recn_imgs.size(-1)` thay vì `max_img_len - style_imgs.size(-1)`.

Vì `style_imgs` và `recn_imgs` có thể có width khác nhau, đây là một bug logic tiềm ẩn ở khâu visualization/sample generation.

Nó không nhất thiết phá train loop chính, nhưng là một bug thật trong baseline hiện tại.

### 18.6. Một số đường code mang tính legacy / interactive

- `eval_demo.py` phụ thuộc `input()` và `plt.show()`.
- Một số path / config phản ánh assumptions của môi trường upstream cũ.
- Có code utility hoặc method ít được dùng trong runtime chính.

### 18.7. DDP có mặt nhưng chưa có workflow đầy đủ

Code có xử lý `local_rank` và `DistributedDataParallel`, nhưng repo hiện không có một launch workflow multi-GPU rõ ràng ở mức docs/script.

## 19. Những hiểu lầm dễ gặp nếu không đọc code

### 19.1. “Train from scratch” trong `gan_iam.yml`

Không có nghĩa là train toàn bộ hệ từ số 0.

Đúng hơn phải hiểu là:

- GAN chính (`G/D/E`) train từ đầu
- auxiliary teacher (`R/W/B`) warm-start từ checkpoint phụ trợ

### 19.2. `deploy_HiGAN+.pth` không phải full training checkpoint

Nó là deploy bundle rút gọn, không phải snapshot đầy đủ của toàn bộ training state.

### 19.3. Sample đẹp sớm khi fine-tune không có nghĩa model học cực nhanh

Thường là vì generator đã warm-start từ deploy checkpoint tốt sẵn.

## 20. Baseline câu hỏi cho phase research kế tiếp

Trước khi nâng cấp model, các câu hỏi baseline nên được chốt từ report này là:

1. Mục tiêu research là cải tiến inference quality, train stability, style controllability, hay khả năng full-scratch reproduction?
2. Có cần đóng kín full pipeline scratch cho `OCR -> WID -> HiGAN+` không?
3. Có giữ nguyên auxiliary-teacher design hiện tại hay thay bằng objective khác?
4. Có tiếp tục giữ BigGAN-style backbone hay thay bằng kiến trúc hiện đại hơn?
5. Có nên sửa discrepancy `lambda_gram` để config phản ánh đúng runtime?
6. Có nên giảm chi phí validation hoặc tách metric pipeline ra khỏi training loop?

## 21. Kết luận

Baseline hiện tại là một fork HiGAN+ đã usable và đã được vá để chạy trên môi trường PyTorch/CUDA hiện đại, có thể smoke test local, fine-tune trên Kaggle, và suy luận headless. Tuy vậy, nó vẫn mang theo một số giới hạn quan trọng của upstream design và một vài mismatch nội bộ của chính fork hiện tại.

Nếu mục tiêu kế tiếp là nâng cấp model hoặc làm research nghiêm túc, có 3 fact nền cần giữ rất rõ:

- Model chính hiện không sống độc lập; nó phụ thuộc mạnh vào `Recognizer`, `WriterIdentifier`, và `StyleBackbone` warm-start để train ổn định theo workflow chuẩn của repo.
- Fine-tune từ `deploy_HiGAN+.pth` và train GAN từ random init là hai trạng thái rất khác nhau, cần tách bạch trong mọi thí nghiệm.
- “Full scratch toàn pipeline” hiện chưa reproducible chỉ bằng setup có sẵn, do mismatch dataset giữa `iam_word` và `iam_word_org`.

Đây là điểm xuất phát chính xác để bước sang phase cải tiến.
