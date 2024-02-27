class config:
    def __init__(self):
        self.train_img_pkl = "./datatrain/faces.pkl"
        self.train_label_pkl = "./datatrain/y_labels.pkl"
        self.val_img_pkl = "./dataval/faces.pkl"
        self.val_label_pkl = "./dataval/y_labels.pkl"
        self.test_img_pkl = "./datatest/faces.pkl"
        self.test_label_pkl = "./datatest/y_labels.pkl"
        self.num_epoch = 20
        self.batch_size = 32
        self.pretrained_path = "./model/model_triplot.h5"
