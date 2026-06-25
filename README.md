# Autoencoder with RBM Pretraining

## 1. Giới thiệu

Project này xây dựng mô hình **Deep Autoencoder** cho bài toán giảm chiều và tái tạo ảnh chữ số viết tay MNIST. Điểm chính của notebook là sử dụng **Restricted Boltzmann Machine (RBM)** để tiền huấn luyện từng tầng, sau đó dùng các trọng số RBM để khởi tạo Autoencoder và fine-tune toàn bộ mô hình end-to-end.

Ngoài mô hình Autoencoder, notebook còn có phần **PCA baseline** kết hợp với **Random Forest Classifier** để so sánh khả năng giảm chiều và phân loại khi dùng PCA với số chiều khác nhau.

## 2. Mục tiêu

- Tải và xử lý dữ liệu MNIST dạng ảnh xám 28x28.
- Làm phẳng ảnh từ kích thước `28 x 28` thành vector `784` chiều.
- Huấn luyện nhiều RBM theo từng tầng để học đặc trưng không giám sát.
- Khởi tạo Deep Autoencoder từ các RBM đã huấn luyện.
- Fine-tune Autoencoder bằng Binary Cross Entropy Loss.
- Hiển thị ảnh gốc và ảnh tái tạo để đánh giá trực quan.
- Vẽ biểu đồ loss trong quá trình fine-tune.
- Chạy PCA baseline để so sánh với biểu diễn 30 chiều của Autoencoder.

## 3. Dataset

Notebook sử dụng **MNIST**, gồm ảnh chữ số viết tay từ 0 đến 9.

Trong phần Autoencoder, dữ liệu được tải bằng `torchvision.datasets.MNIST`:

- Tập train: `60,000` ảnh.
- Mỗi ảnh có kích thước `28 x 28`.
- Sau khi flatten, mỗi ảnh trở thành vector `784` chiều.
- Pixel được đưa về dạng tensor trong khoảng `[0, 1]`.

Trong phần PCA baseline, dữ liệu MNIST được tải bằng `fetch_openml("mnist_784")` từ `scikit-learn`.

## 4. Kiến trúc mô hình

### 4.1. RBM

Notebook định nghĩa lớp `BernoulliRBM` với:

- `n_vis`: số node visible layer.
- `n_hid`: số node hidden layer.
- `W`: ma trận trọng số giữa visible và hidden layer.
- `bv`: bias của visible layer.
- `bh`: bias của hidden layer.

RBM được huấn luyện bằng thuật toán **Contrastive Divergence 1 bước (CD-1)**.

Các RBM được train lần lượt như sau:

| RBM | Kích thước |
|---|---:|
| RBM1 | `784 -> 1000` |
| RBM2 | `1000 -> 500` |
| RBM3 | `500 -> 250` |
| RBM4 | `250 -> 30` |

Sau mỗi RBM, output hidden probability được dùng làm input để huấn luyện RBM tiếp theo.

### 4.2. Deep Autoencoder

Autoencoder có kiến trúc:

```text
Input 784
  -> Encoder: 1000 -> 500 -> 250 -> 30
  -> Decoder: 250 -> 500 -> 1000 -> 784
Output 784
```

Trong đó:

- Các hidden layer dùng hàm kích hoạt `Sigmoid`.
- Code layer có kích thước `30` và là layer tuyến tính.
- Output layer dùng `Sigmoid` để tái tạo ảnh trong khoảng `[0, 1]`.

## 5. Luồng xử lý chính

1. Cài và import thư viện cần thiết.
2. Thiết lập cấu hình như batch size, learning rate, số bước train RBM và số epoch fine-tune.
3. Tải MNIST và flatten ảnh thành vector 784 chiều.
4. Huấn luyện RBM1 trên dữ liệu gốc.
5. Dùng RBM1 để tạo đặc trưng hidden, sau đó huấn luyện RBM2.
6. Lặp lại quá trình trên cho RBM3 và RBM4.
7. Khởi tạo Deep Autoencoder bằng trọng số của 4 RBM.
8. Fine-tune Autoencoder trên toàn bộ tập train bằng `LBFGS` và `BCELoss`.
9. Vẽ ảnh gốc và ảnh tái tạo.
10. Vẽ biểu đồ loss.
11. Chạy PCA baseline và lưu kết quả vào thư mục `artifacts`.

## 6. Cấu hình chính

Các thông số chính trong notebook:

```python
SEED = 0
BATCH = 128
LR_RBM = 0.03
AE_SIZES = (784, 1000, 500, 250, 30)
STEPS_RBM1 = 800
STEPS_RBM2 = 800
STEPS_RBM3 = 800
STEPS_RBM4 = 800
EPOCHS_AE = 30
```

Mô hình tự động dùng GPU nếu có:

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## 7. Cài đặt môi trường

Có thể chạy notebook trên Google Colab, Kaggle hoặc Jupyter Notebook.

Cài các thư viện cần thiết:

```bash
pip install torch torchvision matplotlib numpy pandas scikit-learn joblib
```

## 8. Cách chạy

Mở notebook:

```bash
jupyter notebook "Autoencoder_with_rbm (12).ipynb"
```

Sau đó chạy tuần tự các cell từ trên xuống dưới.

Lưu ý:

- Lần chạy đầu tiên sẽ tải dataset MNIST.
- Quá trình huấn luyện RBM và fine-tune Autoencoder có thể mất thời gian nếu chạy bằng CPU.
- Nếu dùng GPU, thời gian chạy sẽ nhanh hơn đáng kể.

## 9. Kết quả chính trong notebook

### 9.1. RBM pretraining

RBM được train theo từng tầng. Reconstruction BCE của RBM1 giảm rõ rệt từ khoảng `0.6966` xuống khoảng `0.1481`, cho thấy RBM đầu tiên học được biểu diễn tốt hơn từ dữ liệu gốc.

RBM4 cũng giảm reconstruction BCE từ khoảng `0.6933` xuống khoảng `0.4084` sau 800 bước train.

### 9.2. Autoencoder fine-tuning

Sau khi khởi tạo từ 4 RBM, Autoencoder được fine-tune trong 30 epoch bằng `LBFGS` và `BCELoss`.

Loss giảm từ:

```text
Epoch 1:  0.279139
Epoch 30: 0.121713
```

Điều này cho thấy mô hình học được cách tái tạo ảnh MNIST tốt hơn qua quá trình fine-tune.

### 9.3. PCA baseline

Notebook cũng chạy PCA với nhiều số chiều khác nhau và dùng Random Forest để đánh giá accuracy.

Một số kết quả:

| Phương pháp | Số chiều | Accuracy | Explained variance | Reconstruction MSE |
|---|---:|---:|---:|---:|
| Raw scaled pixels + RF | 784 | 0.9721 | - | - |
| PCA + RF | 30 | 0.9564 | 0.7316 | 0.0181 |
| PCA + RF | 40 | 0.9579 | 0.7869 | 0.0143 |
| PCA + RF | 100 | 0.9535 | 0.9150 | 0.0057 |

Kết quả cho thấy PCA với 30 chiều vẫn giữ được accuracy khá cao, phù hợp để so sánh với code layer 30 chiều của Autoencoder.

## 10. File output

Phần PCA baseline có thể tạo thư mục:

```text
artifacts/
```

Trong thư mục này có thể có các file:

```text
pca_experiment_results.csv
pca_grid_results.csv
pca_k_*.joblib
Xte_recon_k*.npy
pca_accuracy_vs_k.png
pca_cum_var_vs_k.png
pca_recon_mse_vs_k.png
scaler.joblib
```

Các file này lưu kết quả PCA, model PCA, ảnh tái tạo và biểu đồ so sánh.

## 11. Ý nghĩa project

Project minh họa cách xây dựng một mô hình giảm chiều dữ liệu ảnh bằng Deep Autoencoder. RBM được dùng như bước pretraining để giúp khởi tạo trọng số tốt hơn trước khi fine-tune toàn bộ Autoencoder. Code layer 30 chiều đóng vai trò biểu diễn nén của ảnh MNIST, trong khi decoder cố gắng tái tạo lại ảnh ban đầu từ biểu diễn nén này.

Phần PCA baseline giúp so sánh cách giảm chiều tuyến tính của PCA với biểu diễn phi tuyến học được từ Autoencoder.

## 12. Ghi chú

- Đây là notebook học thuật/thực nghiệm, chưa được tách thành module Python hoàn chỉnh.
- Nên chạy các cell theo đúng thứ tự để tránh lỗi biến chưa được định nghĩa.
- Nếu chỉ muốn chạy nhanh để kiểm thử, có thể giảm số bước train RBM hoặc số epoch fine-tune.
- Nếu dùng Google Colab, nên bật GPU trong phần `Runtime > Change runtime type > GPU`.
