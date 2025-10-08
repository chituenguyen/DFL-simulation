# TÓM TẮT DỰ ÁN - DFL BASIC SIMULATION

## 📋 Tổng quan
**Tên dự án:** Decentralized Federated Learning (DFL) Basic Simulation
**Phiên bản:** 1.0.0
**Mô tả:** Mô phỏng hệ thống Học máy liên kết phi tập trung (Decentralized Federated Learning) sử dụng mô hình ResNet-18 trên tập dữ liệu CIFAR-10

## 🎯 Mục đích
- Mô phỏng quá trình huấn luyện mô hình phân tán trên nhiều node
- Thử nghiệm các thuật toán tổng hợp (aggregation) trong Federated Learning
- Hỗ trợ phân vùng dữ liệu IID và Non-IID
- Hỗ trợ nhiều topology mạng khác nhau (ring, mesh, star, random)

## 🏗️ Cấu trúc dự án

```
dfl_basic_simulation/
├── config/                  # File cấu hình
│   ├── config.yaml         # Config chính
│   └── config_test.yaml    # Config test
├── src/                    # Source code chính
│   ├── aggregation/        # Thuật toán tổng hợp (FedAvg)
│   ├── communication/      # Giao thức truyền thông P2P
│   ├── data/              # Xử lý dữ liệu và phân vùng
│   ├── models/            # Định nghĩa mô hình (ResNet-18)
│   ├── node/              # Quản lý node và base node
│   └── utils/             # Công cụ hỗ trợ (logger, metrics, visualizer)
├── experiments/            # Các thử nghiệm
│   ├── basic_dfl.py       # DFL cơ bản
│   ├── enhanced_dfl.py    # DFL nâng cao
│   ├── checkpoint_demo.py # Demo checkpoint
│   ├── single_node_demo.py # Demo 1 node
│   └── run_experiment.py  # Script chạy experiment
├── tests/                 # Unit tests
├── scripts/               # Scripts hỗ trợ
├── data/                  # Dữ liệu CIFAR-10
├── results/               # Kết quả thực nghiệm
├── checkpoints/           # Model checkpoints
└── notebooks/             # Jupyter notebooks

```

## 🚀 Cách chạy

### 1. Cài đặt môi trường

```bash
# Tạo virtual environment
python -m venv venv

# Kích hoạt virtual environment
# MacOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Cài đặt dependencies
pip install -r requirements.txt
```

### 2. Chạy thử nghiệm cơ bản

```bash
# Chạy với cấu hình mặc định
python experiments/run_experiment.py

# Chạy với file config tùy chỉnh
python experiments/run_experiment.py --config config/config.yaml

# Chạy với số rounds tùy chỉnh
python experiments/run_experiment.py --rounds 10

# Chỉ định device (cuda/cpu/mps)
python experiments/run_experiment.py --device mps
```

### 3. Chạy các demo khác

```bash
# Demo single node
python experiments/single_node_demo.py

# Demo checkpoint
python experiments/checkpoint_demo.py

# DFL cơ bản
python experiments/basic_dfl.py

# DFL nâng cao
python experiments/enhanced_dfl.py
```

### 4. Chạy tests

```bash
# Chạy tất cả tests
pytest tests/

# Chạy test với coverage
pytest tests/ --cov=src --cov-report=html
```

## ⚙️ Cấu hình chính

**Model:**
- ResNet-18 cho CIFAR-10 (10 classes)

**Dữ liệu:**
- Dataset: CIFAR-10
- Batch size: 128
- Phân vùng: IID / Non-IID (Dirichlet alpha=0.5)

**Training:**
- Số rounds: 50 (mặc định)
- Local epochs: 5
- Learning rate: 0.01
- Optimizer: SGD với momentum 0.9

**Topology:**
- Kiểu: ring, mesh, star, random
- Số neighbors: 2 (mặc định)

**Aggregation:**
- Thuật toán: FedAvg
- Weighted by dataset size
- Hỗ trợ momentum và adaptive weighting

## 📊 Kết quả

Kết quả được lưu trong thư mục `results/`:
- Logs: `results/logs/`
- Models: `results/models/`
- Tensorboard: Có hỗ trợ

## 🛠️ Công nghệ sử dụng

- **Framework:** PyTorch >= 2.0.0
- **Deep Learning:** torchvision, ResNet-18
- **Data Processing:** numpy, pandas
- **Visualization:** matplotlib, seaborn, tensorboard
- **Testing:** pytest
- **Code Quality:** black, flake8, isort

## 📝 Tính năng chính

1. ✅ Mô phỏng DFL với nhiều node
2. ✅ Hỗ trợ phân vùng dữ liệu IID/Non-IID
3. ✅ Nhiều topology mạng (ring, mesh, star, random)
4. ✅ Thuật toán FedAvg aggregation
5. ✅ Checkpoint và resume training
6. ✅ Metrics tracking và visualization
7. ✅ TensorBoard integration
8. ✅ Mô phỏng latency và bandwidth
9. ✅ Privacy-preserving (chuẩn bị cho differential privacy)

## 👤 Tác giả
Senior Developer

## 📄 License
MIT License
