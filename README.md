# Facial expression recognition
> SGU 2026

> Running in Kaggle: https://www.kaggle.com/code/doduyquynii/sgu-2026-fer/edit (Team Private)

> Link dataset: https://drive.google.com/drive/folders/1VrCqZw4M8f7w5vzXWtYmdyLRKYutH7Oh?usp=sharing


## Yêu cầu: 
1. Mô hình đi từ kiến trúc đơn giản nhất CNN gốc đi lên VGG, ResNet hay ResMaskingNet,...
2. Template rõ ràng các phần
3. Show:
    - 10 ảnh đúng, tại sao nó đúng?
    - 10 ảnh sai, phân tích nguyên nhân sai, đưa ra phương pháp khắc phục 
4. Model code trên IDE(local) và chạy trên Kaggle (có thể chạy local), viết code chạy đồng thời cho cả 2, mỗi lần chạy ở đầu thì chỉ cần thay đổi config
5. Log thực nghiệm và theo dõi trên Wandb

## Cấu trúc:
```
sgu-2026-facial-expression-recognition/
│
├── configs/                     #    Tất cả config (YAML)
│   ├── base.yaml                #    Config mặc định chung
│   ├── simple_cnn.yaml        #    Config cho CNN cơ bản
│   ├── vgg19.yaml               #    Config cho VGG19
│   ├── resnet34.yaml            #    Config cho ResNet34
│   └── env.yaml                 #    Cấu hình môi trường (local / kaggle)
│
├── data/                        #    Dữ liệu (KHÔNG push lên Git)
│   └── fer13-split/
│       ├── train.csv              #    Training
│       ├── val.csv                #    PublicTest
│       └── test.csv               #    PrivateTest
│
├── src/                         #    Source code chính
│   ├── __init__.py
│   │
│   ├── data/                    #    Data pipeline
│   │   ├── __init__.py
│   │   ├── dataset.py           
│   │   ├── transforms.py        #    Data augmentation & preprocessing
│   │   └── dataloader.py        #    Hàm tạo DataLoader (train/val/test)
│   │
│   ├── models/                  #    Tất cả kiến trúc mô hình
│   │   ├── __init__.py          #    Registry: get_model(name) -> model
│   │   ├── simple_cnn.py        #    CNN baseline (3-5 Conv layers)
│   │   ├── vgg.py               #    VGG11/16/19 (fine-tune từ pretrained)
│   │   ├── resnet.py            #    ResNet18/34/50 (fine-tune)
│   │   └── ....py               #    another ()
│   │
│   ├── training/                #    Training pipeline
│   │   ├── __init__.py
│   │   ├── trainer.py           #    Trainer class (train/val loop)
│   │   ├── optimizer.py         #    Hàm tạo optimizer & scheduler
│   │   └── losses.py            #    Loss functions (CrossEntropy, FocalLoss,...)
│   │
│   ├── evaluation/              #    Đánh giá & phân tích
│   │   ├── __init__.py
│   │   ├── metrics.py           #    Accuracy, F1, Confusion Matrix
│   │   ├── evaluator.py         #    Chạy evaluation trên test set
│   │   └── error_analysis.py    #    Phân tích 10 đúng / 10 sai
│   │
│   └── utils/                   #    Utils chung 
│       ├── __init__.py
│       ├── config.py            #    Load & merge YAML configs
│       ├── logger.py            #    Setup WandB + local logging
│       ├── checkpoint.py        #    Save/load model checkpoints
│       ├── seed.py              #    Set random seed cho reproducibility
│       └── visualization.py     #    Plot confusion matrix, samples,...
│
├── notebooks/                   #    Jupyter Notebooks (EDA & demo)
│   ├── 01_eda.ipynb             
│   ├── 02_training_demo.ipynb   #    Demo training trên Kaggle
│   └── 03_error_analysis.ipynb  #    Phân tích lỗi có visualization
│
├── scripts/                     #    Scripts chạy nhanh
│   ├── prepare_data.py          #    Convert CSV → image folders
│   ├── train.py                 #    Entry point: training
│   ├── evaluate.py              #    Entry point: evaluation
│   └── analyze_errors.py        #    Entry point: phân tích lỗi
│
├── outputs/                     #    Kết quả (KHÔNG push lên Git)
│   ├── checkpoints/             #    Model weights (.pth)
│   ├── logs/                    #    Training logs
│   └── figures/                 #    Confusion matrix, sample images,...
│
├── tests/                       #    Unit tests
│   ├── test_dataset.py
│   ├── test_models.py
│   └── test_trainer.py
│
├── .gitignore
├── requirements.txt             #    Dependencies
└── README.md                    #    Hướng dẫn tổng quan
```
