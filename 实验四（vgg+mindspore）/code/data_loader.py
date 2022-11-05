import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C


def creat_dataset(dir, img_size=[224, 224], batch_size=32):
    data = ds.ImageFolderDataset(dir,
                                 class_indexing={"daisy": 0,
                                                 "dandelion": 1,
                                                 "roses": 2,
                                                 "sunflowers": 3,
                                                 "tulips": 4})
    trans = CV.RandomCropDecodeResize(img_size, scale=(0.08, 1.0), ratio=(0.75, 1.33))
    ds_type = C.TypeCast(ms.float32)
    data = data.map(input_columns="image", operations=trans)
    data = data.map(input_columns="image", operations=CV.HWC2CHW())
    data = data.map(input_columns="image", operations=ds_type)
    data = data.shuffle(buffer_size=batch_size * 10)
    data = data.batch(batch_size, drop_remainder=True)
    data = data.map(input_columns="image", operations=CV.Rescale(1.0 / 255, 0))
    return data


if __name__ == '__main__':
    dataset = creat_dataset("../data/flower_photos/train")
    for data, labels in dataset:
        print(data, labels)
        break
