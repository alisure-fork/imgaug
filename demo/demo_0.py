import os
import time
import imgaug
import imageio
import numpy as np
from imgaug import multicore
import matplotlib.pyplot as plt
from alisuretool.Tools import Tools
from imgaug.augmentables.polys import Polygon
from imgaug.augmentables.kps import Keypoint
import imgaug.parameters as imgaug_parameters
import imgaug.augmenters as imgaug_augmenters
from imgaug.augmentables.heatmaps import HeatmapsOnImage
from imgaug.augmentables.batches import UnnormalizedBatch
from imgaug.augmentables.segmaps import SegmentationMapOnImage
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
from imgaug.augmentables.polys import Polygon, PolygonsOnImage
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


# 0
class Help(object):

    """
    https://github.com/aleju/imgaug
    https://github.com/aleju/imgaug-doc
    https://imgaug.readthedocs.io/en/latest/
    https://nbviewer.jupyter.org/github/aleju/imgaug-doc/tree/master/notebooks/
    """

    pass


# A1: 基本操作
class LoadAndAugmentAnImage(object):

    ##################################################################################################
    # https://nbviewer.jupyter.org/github/aleju/imgaug-doc/blob/master/notebooks/A01%20-%20Load%20and%20Augment%20an%20Image.ipynb
    # Load and Augment an Image
    ##################################################################################################

    @staticmethod
    def demo_1_read_image(result_file="./image/Lenna_test_image.png", is_show=False,
                          origin_file="https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png"):
        if os.path.exists(result_file):
            image = imageio.imread(result_file)
        else:
            image = imageio.imread(origin_file)
            result_file = Tools.new_dir(result_file)
            imageio.imwrite(result_file, image)
            pass

        if is_show:
            imgaug.imshow(image)
            pass

        return image

    @classmethod
    def demo_2_augmenters(cls):
        image = cls.demo_1_read_image()

        rotate = imgaug_augmenters.Affine(rotate=(-25, 25))
        image_aug = rotate.augment_image(image)

        Tools.print("Augmented:")
        imgaug.imshow(image_aug)

        images = [image, image, image, image]
        images_aug = rotate.augment_images(images)

        Tools.print("Augmented Batch:")
        imgaug.imshow(np.hstack(images_aug))
        pass

    @classmethod
    def demo_3_sequential(cls):
        image = cls.demo_1_read_image()

        seq = imgaug_augmenters.Sequential([
            imgaug_augmenters.Affine(rotate=(-25, 25)),
            imgaug_augmenters.AdditiveGaussianNoise(scale=(10, 60)),
            imgaug_augmenters.Crop(percent=(0, 0.2))
        ])

        images = [image, image, image, image]
        images_aug = seq.augment_images(images)

        Tools.print("Augmented Batch/Sequential:")
        imgaug.imshow(np.hstack(images_aug))
        pass

    @classmethod
    def demo_4_sequential_random_order(cls):
        image = cls.demo_1_read_image()

        seq = imgaug_augmenters.Sequential([
            imgaug_augmenters.Affine(rotate=(-25, 25)),
            imgaug_augmenters.AdditiveGaussianNoise(scale=(30, 90)),
            imgaug_augmenters.Crop(percent=(0, 0.4))
        ], random_order=True)

        images_aug = [seq.augment_image(image) for _ in range(8)]

        Tools.print("Augmented Sequential:")
        imgaug.imshow(imgaug.draw_grid(images_aug, cols=4, rows=2))
        pass

    @classmethod
    def demo_5_sequential_random_order_different_sizes(cls):
        seq = imgaug_augmenters.Sequential([
            imgaug_augmenters.CropAndPad(percent=(-0.2, 0.2), pad_mode="edge"),  # crop and pad images
            imgaug_augmenters.AddToHueAndSaturation((-60, 60)),  # change their color
            imgaug_augmenters.ElasticTransformation(alpha=90, sigma=9),  # water-like effect
            imgaug_augmenters.CoarseDropout((0.01, 0.1), size_percent=0.01)  # set large image areas to zero
        ], random_order=True)

        # load images with different sizes
        images_different_sizes = [
            cls.demo_1_read_image(
                result_file="./image/BRACHYLAGUS_IDAHOENSIS.jpg",
                origin_file="https://upload.wikimedia.org/wikipedia/commons/e/ed/BRACHYLAGUS_IDAHOENSIS.jpg"),
            cls.demo_1_read_image(
                result_file="./image/Southern_swamp_rabbit_baby.jpg",
                origin_file="https://upload.wikimedia.org/wikipedia/commons/c/c9/Southern_swamp_rabbit_baby.jpg"),
            cls.demo_1_read_image(
                result_file="./image/Lower_Keys_marsh_rabbit.jpg",
                origin_file="https://upload.wikimedia.org/wikipedia/commons/9/9f/Lower_Keys_marsh_rabbit.jpg")
        ]

        # augment them as one batch
        images_aug = seq.augment_images(images_different_sizes)

        # visualize the results
        Tools.print("Image 0 (input shape: %s, output shape: %s)" % (images_different_sizes[0].shape, images_aug[0].shape))
        imgaug.imshow(np.hstack([images_different_sizes[0], images_aug[0]]))

        Tools.print("Image 1 (input shape: %s, output shape: %s)" % (images_different_sizes[1].shape, images_aug[1].shape))
        imgaug.imshow(np.hstack([images_different_sizes[1], images_aug[1]]))

        Tools.print("Image 2 (input shape: %s, output shape: %s)" % (images_different_sizes[2].shape, images_aug[2].shape))
        imgaug.imshow(np.hstack([images_different_sizes[2], images_aug[2]]))
        pass

    ##################################################################################################

    pass


# A2： 随机模式or相同模式
class StochasticAndDeterministicAugmentation(object):

    ##################################################################################################
    # https://nbviewer.jupyter.org/github/aleju/imgaug-doc/blob/master/notebooks/A02%20-%20Stochastic%20and%20Deterministic%20Augmentation.ipynb
    # Stochastic and Deterministic Augmentation
    ##################################################################################################

    @staticmethod
    def demo_6_stochastic_and_deterministic_mode():
        """
        The result for stochastic mode should differ between the batches,
         while the one in deterministic mode should be the same for both batches.
        :return:
        """
        aug = imgaug_augmenters.Affine(translate_px=(-30, 30), rotate=(-20, 20), cval=255)
        image = imgaug.quokka(size=0.15)
        batches = [[image] * 3, [image] * 3]

        # augment in stochastic mode: differ between the batches
        images_stochastic = [aug.augment_images(batch) for batch in batches]

        # augment in deterministic mode: the same for both batches
        aug_det = aug.to_deterministic()
        images_deterministic = [aug_det.augment_images(batch) for batch in batches]

        # visualize
        whitespace = np.full(image.shape, 255, dtype=np.uint8)
        imgaug.imshow(
            imgaug.draw_grid(
                images_stochastic[0] + [whitespace] + images_stochastic[1] +  # first row
                images_deterministic[0] + [whitespace] + images_deterministic[1],  # second row
                rows=2, cols=2 * 3
            )
        )
        pass

    @staticmethod
    def demo_7_stochastic_and_deterministic_mode_keypoints():
        image = imgaug.quokka(size=0.25)
        keypoints = imgaug.quokka_keypoints(size=0.25)
        imgaug.imshow(np.hstack([image, keypoints.draw_on_image(image)]))

        BATCH_SIZE = 4
        images_batch = [image] * BATCH_SIZE
        keypoints_batch = [keypoints] * BATCH_SIZE

        aug = imgaug_augmenters.Affine(rotate=[0, 15, 30, 45, 60])

        ###################################################################################
        # 1.stochastic mode
        # images_aug1 = aug.augment_images(images_batch)
        # images_aug2 = aug.augment_images(images_batch)
        # images_aug3 = aug.augment_images(images_batch)
        # keypoints_aug1 = aug.augment_keypoints(keypoints_batch)
        # keypoints_aug2 = aug.augment_keypoints(keypoints_batch)
        # keypoints_aug3 = aug.augment_keypoints(keypoints_batch)
        ###################################################################################
        # 2.deterministic mode
        # aug = aug.to_deterministic()  # <- this changed
        # images_aug1 = aug.augment_images(images_batch)
        # images_aug2 = aug.augment_images(images_batch)
        # images_aug3 = aug.augment_images(images_batch)
        # keypoints_aug1 = aug.augment_keypoints(keypoints_batch)
        # keypoints_aug2 = aug.augment_keypoints(keypoints_batch)
        # keypoints_aug3 = aug.augment_keypoints(keypoints_batch)
        ###################################################################################
        # 3.Different augmentations between batches and images, but the same one for each image and the keypoints on it.
        aug = aug.to_deterministic()
        images_aug1 = aug.augment_images(images_batch)
        keypoints_aug1 = aug.augment_keypoints(keypoints_batch)
        aug = aug.to_deterministic()
        images_aug2 = aug.augment_images(images_batch)
        keypoints_aug2 = aug.augment_keypoints(keypoints_batch)
        aug = aug.to_deterministic()
        images_aug3 = aug.augment_images(images_batch)
        keypoints_aug3 = aug.augment_keypoints(keypoints_batch)
        ###################################################################################

        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 8))
        axes[0].imshow(np.hstack([kp_i.draw_on_image(im_i) for im_i, kp_i in zip(images_aug1, keypoints_aug1)]))
        axes[1].imshow(np.hstack([kp_i.draw_on_image(im_i) for im_i, kp_i in zip(images_aug2, keypoints_aug2)]))
        axes[2].imshow(np.hstack([kp_i.draw_on_image(im_i) for im_i, kp_i in zip(images_aug3, keypoints_aug3)]))
        for i in range(3):
            axes[i].set_title("Batch %d" % (i + 1,))
            axes[i].axis("off")
        plt.show()
        pass

    ##################################################################################################

    pass


# A3： 多CPU
class AugmentationOnMultipleCPUCores(object):

    ##################################################################################################
    # https://nbviewer.jupyter.org/github/aleju/imgaug-doc/blob/master/notebooks/A03%20-%20Multicore%20Augmentation.ipynb
    # Augmentation on multiple CPU cores
    ##################################################################################################

    """
    Conclusion

    So to use multicore augmentation with imgaug just do the following:

        * Convert your data to instances of imgaug.Batch. Make sure that corresponding data has the same list
            index within the batch, e.g. images and their corresponding keypoints.
        * Call augmenter.augment_batches(batches, background=True). This returns a generator.
        * Use augmenter.pool([processes], [maxtasksperchild], [seed]) if you need more control or
            want to use a generator as input. Call pool.map_batches(list) or pool.imap_batches(generator) on the pool.
        * Avoid implementing your own multicore system or using another library for that as it is easy to mess up.
    """

    @staticmethod
    def demo_8_multicore(batch_size=16, num_batches=100):
        image = imgaug.quokka_square(size=(256, 256))
        images = [np.copy(image) for _ in range(batch_size)]

        batches = [UnnormalizedBatch(images=images) for _ in range(num_batches)]

        aug = imgaug_augmenters.Sequential([
            imgaug_augmenters.PiecewiseAffine(scale=0.05, nb_cols=6, nb_rows=6),  # very slow
            imgaug_augmenters.Fliplr(0.5),  # very fast
            imgaug_augmenters.CropAndPad(px=(-10, 10))  # very fast
        ])

        # singlecore aug
        time_start = time.time()
        batches_aug = list(aug.augment_batches(batches, background=False))  # list() converts generator to list
        time_end = time.time()

        Tools.print("Augmentation done in %.2fs" % (time_end - time_start,))
        imgaug.imshow(batches_aug[0].images_aug[0])

        # multicore aug
        time_start = time.time()
        batches_aug = list(aug.augment_batches(batches, background=True))  # background=True for multicore aug
        time_end = time.time()

        Tools.print("Augmentation done in %.2fs" % (time_end - time_start,))
        imgaug.imshow(batches_aug[0].images_aug[0])
        pass

    @staticmethod
    def demo_9_multicore_non_image_data_keypoints(batch_size=16, num_batches=100):
        image = imgaug.quokka(size=0.2)
        images = [np.copy(image) for _ in range(batch_size)]
        keypoints = imgaug.quokka_keypoints(size=0.2)
        keypoints = [keypoints.deepcopy() for _ in range(batch_size)]

        batches = [UnnormalizedBatch(images=images, keypoints=keypoints) for _ in range(num_batches)]

        aug = imgaug_augmenters.Sequential([
            imgaug_augmenters.PiecewiseAffine(scale=0.05, nb_cols=6, nb_rows=6),  # very slow
            imgaug_augmenters.Fliplr(0.5),  # very fast
            imgaug_augmenters.CropAndPad(px=(-10, 10))  # very fast
        ])

        time_start = time.time()
        batches_aug = list(aug.augment_batches(batches, background=True))  # background=True for multicore aug
        time_end = time.time()

        print("Augmentation done in %.2fs" % (time_end - time_start,))
        imgaug.imshow(batches_aug[0].keypoints_aug[0].draw_on_image(batches_aug[0].images_aug[0]))
        pass

    @staticmethod
    def demo_10_multicore_pool(batch_size=16, num_batches=100):
        image = imgaug.quokka(size=0.2)
        images = [np.copy(image) for _ in range(batch_size)]

        batches = [UnnormalizedBatch(images=images) for _ in range(num_batches)]

        aug = imgaug_augmenters.Sequential([
            imgaug_augmenters.PiecewiseAffine(scale=0.05, nb_cols=6, nb_rows=6),  # very slow
            imgaug_augmenters.Fliplr(0.5),  # very fast
            imgaug_augmenters.CropAndPad(px=(-10, 10))  # very fast
        ])

        ################################################################################################
        # control the number of used CPU cores or the random number seed
        # 可指定使用的CPU数量，每个子进程处理的任务数量， 随机种子
        time_start = time.time()
        with aug.pool(processes=-1, maxtasksperchild=20, seed=1) as pool:
            batches_aug = pool.map_batches(batches)
        time_end = time.time()
        Tools.print("Augmentation done in %.2fs" % (time_end - time_start,))
        imgaug.imshow(batches_aug[0].images_aug[0])
        ################################################################################################

        ################################################################################################
        # 同上
        time_start = time.time()
        with imgaug.multicore.Pool(aug, processes=-1, maxtasksperchild=20, seed=1) as pool:
            batches_aug = pool.map_batches(batches)
        time_end = time.time()
        Tools.print("Augmentation done in %.2fs" % (time_end - time_start,))
        imgaug.imshow(batches_aug[0].images_aug[0])
        ################################################################################################
        pass

    @staticmethod
    def demo_11_multicore_using_pool_with_generators(batch_size=16, num_batches=100):

        def create_generator(lst):
            for list_entry in lst:
                yield list_entry
                pass
            pass

        image = imgaug.quokka(size=0.2)
        images = [np.copy(image) for _ in range(batch_size)]

        batches = [UnnormalizedBatch(images=images) for _ in range(num_batches)]

        aug = imgaug_augmenters.Sequential([
            imgaug_augmenters.PiecewiseAffine(scale=0.05, nb_cols=6, nb_rows=6),  # very slow
            imgaug_augmenters.Fliplr(0.5),  # very fast
            imgaug_augmenters.CropAndPad(px=(-10, 10))  # very fast
        ])

        my_generator = create_generator(batches)

        with aug.pool(processes=-1, seed=1) as pool:
            batches_aug = pool.imap_batches(my_generator)

            for i, batch_aug in enumerate(batches_aug):
                Tools.print("{}".format(i))
                if i == 0:
                    imgaug.imshow(batch_aug.images_aug[0])
                    # do something else with the batch here
                    pass
                pass
            pass

        pass

    @staticmethod
    def demo_12_multicore_using_pool_with_generators_output_buffer_size(batch_size=16, num_batches=100):

        def create_generator(lst):
            for list_entry in lst:
                Tools.print("Loading next unaugmented batch...")
                yield list_entry
                pass
            pass

        image = imgaug.quokka(size=0.2)
        images = [np.copy(image) for _ in range(batch_size)]

        batches = [UnnormalizedBatch(images=images) for _ in range(num_batches)]

        aug = imgaug_augmenters.Sequential([
            imgaug_augmenters.PiecewiseAffine(scale=0.05, nb_cols=6, nb_rows=6),  # very slow
            imgaug_augmenters.Fliplr(0.5),  # very fast
            imgaug_augmenters.CropAndPad(px=(-10, 10))  # very fast
        ])

        my_generator = create_generator(batches)

        with aug.pool(processes=-1, seed=1) as pool:
            batches_aug = pool.imap_batches(my_generator, output_buffer_size=5)

            Tools.print("Requesting next augmented batch...")
            for i, batch_aug in enumerate(batches_aug):
                # sleep here for a while to simulate a slowly training model
                time.sleep(0.1)

                if i < len(batches) - 1:
                    Tools.print("Requesting next augmented batch... {}".format(i))
                    pass
                pass
            pass

        pass

    pass


# B1
class AugmentKeypointsLandmarks(object):

    ##################################################################################################
    # https://nbviewer.jupyter.org/github/aleju/imgaug-doc/blob/master/notebooks/B01%20-%20Augment%20Keypoints.ipynb
    # Augment Keypoints/Landmarks
    ##################################################################################################

    @staticmethod
    def demo_13_keypoint():
        image = LoadAndAugmentAnImage.demo_1_read_image(
            result_file="./image/Macropus_rufogriseus_rufogriseus_Bruny.jpg",
            origin_file="https://upload.wikimedia.org/wikipedia/commons/e/e6/Macropus_rufogriseus_rufogriseus_Bruny.jpg")
        image = imgaug.imresize_single_image(image, (389, 259))

        kps = [
            Keypoint(x=99, y=81),  # left eye (from camera perspective)
            Keypoint(x=125, y=80),  # right eye
            Keypoint(x=112, y=102),  # nose
            Keypoint(x=102, y=210),  # left paw
            Keypoint(x=127, y=207)  # right paw
        ]
        kpsoi = KeypointsOnImage(kps, shape=image.shape)

        imgaug.imshow(kpsoi.draw_on_image(image, size=7))

        Tools.print(kpsoi.keypoints)
        pass

    @staticmethod
    def demo_14_keypoint_aug():
        image = LoadAndAugmentAnImage.demo_1_read_image(
            result_file="./image/Macropus_rufogriseus_rufogriseus_Bruny.jpg",
            origin_file="https://upload.wikimedia.org/wikipedia/commons/e/e6/Macropus_rufogriseus_rufogriseus_Bruny.jpg")
        image = imgaug.imresize_single_image(image, (389, 259))

        kps = [
            Keypoint(x=99, y=81),  # left eye (from camera perspective)
            Keypoint(x=125, y=80),  # right eye
            Keypoint(x=112, y=102),  # nose
            Keypoint(x=102, y=210),  # left paw
            Keypoint(x=127, y=207)  # right paw
        ]
        kpsoi = KeypointsOnImage(kps, shape=image.shape)

        seq = imgaug_augmenters.Sequential([
            imgaug_augmenters.Affine(translate_px={"x": (10, 30)}, rotate=(-10, 10)),
            imgaug_augmenters.AddToHueAndSaturation((-50, 50))  # color jitter, only affects the image
        ])
        image_aug, kpsoi_aug = seq(image=image, keypoints=kpsoi)

        imgaug.imshow(np.hstack([kpsoi.draw_on_image(image, size=7), kpsoi_aug.draw_on_image(image_aug, size=7)]))
        pass

    @staticmethod
    def demo_15_keypoint_larger():
        image = LoadAndAugmentAnImage.demo_1_read_image(
            result_file="./image/Macropus_rufogriseus_rufogriseus_Bruny.jpg",
            origin_file="https://upload.wikimedia.org/wikipedia/commons/e/e6/Macropus_rufogriseus_rufogriseus_Bruny.jpg")
        image = imgaug.imresize_single_image(image, (389, 259))

        kps = [
            Keypoint(x=99, y=81),  # left eye (from camera perspective)
            Keypoint(x=125, y=80),  # right eye
            Keypoint(x=112, y=102),  # nose
            Keypoint(x=102, y=210),  # left paw
            Keypoint(x=127, y=207)  # right paw
        ]
        kpsoi = KeypointsOnImage(kps, shape=image.shape)

        image_larger = imgaug.imresize_single_image(image, 2.0)

        Tools.print("Small image %s with keypoints optimized for the size:" % (image.shape,))
        imgaug.imshow(kpsoi.draw_on_image(image, size=7))

        Tools.print("Large image %s with keypoints optimized for the small image size:" % (image_larger.shape,))
        imgaug.imshow(kpsoi.draw_on_image(image_larger, size=7))

        Tools.print("Large image %s with keypoints projected onto that size:" % (image_larger.shape,))
        imgaug.imshow(kpsoi.on(image_larger).draw_on_image(image_larger, size=7))
        pass

    @staticmethod
    def demo_16_keypoint_shift():
        image = LoadAndAugmentAnImage.demo_1_read_image(
            result_file="./image/Macropus_rufogriseus_rufogriseus_Bruny.jpg",
            origin_file="https://upload.wikimedia.org/wikipedia/commons/e/e6/Macropus_rufogriseus_rufogriseus_Bruny.jpg")
        image = imgaug.imresize_single_image(image, (389, 259))

        kps = [
            Keypoint(x=99, y=81),  # left eye (from camera perspective)
            Keypoint(x=125, y=80),  # right eye
            Keypoint(x=112, y=102),  # nose
            Keypoint(x=102, y=210),  # left paw
            Keypoint(x=127, y=207)  # right paw
        ]
        kpsoi = KeypointsOnImage(kps, shape=image.shape)

        image_pad = imgaug.pad(image, left=100)
        kpsoi_pad = kpsoi.shift(x=100)
        imgaug.imshow(kpsoi_pad.draw_on_image(image_pad, size=7))
        pass

    @staticmethod
    def demo_17_keypoint_draw():
        image = LoadAndAugmentAnImage.demo_1_read_image(
            result_file="./image/Macropus_rufogriseus_rufogriseus_Bruny.jpg",
            origin_file="https://upload.wikimedia.org/wikipedia/commons/e/e6/Macropus_rufogriseus_rufogriseus_Bruny.jpg")
        image = imgaug.imresize_single_image(image, (389, 259))

        kps = [
            Keypoint(x=99, y=81),  # left eye (from camera perspective)
            Keypoint(x=125, y=80),  # right eye
            Keypoint(x=112, y=102),  # nose
            Keypoint(x=102, y=210),  # left paw
            Keypoint(x=127, y=207)  # right paw
        ]
        kpsoi = KeypointsOnImage(kps, shape=image.shape)

        # 改变大小
        imgaug.imshow(np.hstack([
            kpsoi.draw_on_image(image, size=1),
            kpsoi.draw_on_image(image, size=3),
            kpsoi.draw_on_image(image, size=5),
            kpsoi.draw_on_image(image, size=7)
        ]))

        # 改变颜色
        imgaug.imshow(np.hstack([
            kpsoi.draw_on_image(image, size=5, color=(0, 255, 0)),
            kpsoi.draw_on_image(image, size=5, color=(0, 0, 255)),
            kpsoi.draw_on_image(image, size=5, color=(255, 128, 255)),
            kpsoi.draw_on_image(image, size=5, color=(255, 255, 255))
        ]))

        # 再同一张图上画多个
        image_draw = np.copy(image)
        kpsoi.draw_on_image(image_draw, size=5, color=(0, 255, 0), copy=False)
        kpsoi.shift(x=-70).draw_on_image(image_draw, size=5, color=(255, 255, 255), copy=False)
        kpsoi.shift(x=70).draw_on_image(image_draw, size=5, color=(0, 0, 0), copy=False)
        imgaug.imshow(image_draw)

        # 不同的点使用不同的颜色
        colors = [(0, 255, 0), (255, 255, 255), (128, 255, 64), (128, 64, 255), (128, 128, 0)]
        image_drawn = np.copy(image)
        for kp, color in zip(kpsoi.keypoints, colors):
            image_drawn = kp.draw_on_image(image_drawn, color=color, size=9, copy=False)
        imgaug.imshow(image_drawn)
        pass

    pass


# B2
class AugmentBoundingBoxes(object):

    ##################################################################################################
    # https://nbviewer.jupyter.org/github/aleju/imgaug-doc/blob/master/notebooks/B02%20-%20Augment%20Bounding%20Boxes.ipynb
    # Augment Bounding Boxes
    ##################################################################################################

    @staticmethod
    def demo_18_bounding_box():
        image = LoadAndAugmentAnImage.demo_1_read_image(
            result_file="./image/Yellow-headed_caracara_Milvago_chimachima_on_capybara_Hydrochoeris_hydrochaeris.JPG",
            origin_file="https://upload.wikimedia.org/wikipedia/commons/8/8e/Yellow-headed_caracara_%28Milvago_chimachima%29_on_capybara_%28Hydrochoeris_hydrochaeris%29.JPG")
        image = imgaug.imresize_single_image(image, (298, 447))

        bbs = BoundingBoxesOnImage([
            BoundingBox(x1=0.2 * 447, x2=0.85 * 447, y1=0.3 * 298, y2=0.95 * 298),
            BoundingBox(x1=0.4 * 447, x2=0.65 * 447, y1=0.1 * 298, y2=0.4 * 298)
        ], shape=image.shape)

        imgaug.imshow(bbs.draw_on_image(image, size=2))
        pass

    @staticmethod
    def demo_19_bounding_box_aug():
        image = LoadAndAugmentAnImage.demo_1_read_image(
            result_file="./image/Yellow-headed_caracara_Milvago_chimachima_on_capybara_Hydrochoeris_hydrochaeris.JPG",
            origin_file="https://upload.wikimedia.org/wikipedia/commons/8/8e/Yellow-headed_caracara_%28Milvago_chimachima%29_on_capybara_%28Hydrochoeris_hydrochaeris%29.JPG")
        image = imgaug.imresize_single_image(image, (298, 447))

        bbs = BoundingBoxesOnImage([
            BoundingBox(x1=0.2 * 447, x2=0.85 * 447, y1=0.3 * 298, y2=0.95 * 298),
            BoundingBox(x1=0.4 * 447, x2=0.65 * 447, y1=0.1 * 298, y2=0.4 * 298)
        ], shape=image.shape)

        seq = imgaug_augmenters.Sequential([
            imgaug_augmenters.GammaContrast(1.5),
            imgaug_augmenters.Affine(translate_percent={"x": 0.1}, scale=0.8)
        ])

        image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)

        imgaug.imshow(bbs_aug.draw_on_image(image_aug, size=2))
        pass

    @staticmethod
    def demo_20_bounding_box_rotate():
        image = LoadAndAugmentAnImage.demo_1_read_image(
            result_file="./image/Yellow-headed_caracara_Milvago_chimachima_on_capybara_Hydrochoeris_hydrochaeris.JPG",
            origin_file="https://upload.wikimedia.org/wikipedia/commons/8/8e/Yellow-headed_caracara_%28Milvago_chimachima%29_on_capybara_%28Hydrochoeris_hydrochaeris%29.JPG")
        image = imgaug.imresize_single_image(image, (298, 447))

        bbs = BoundingBoxesOnImage([
            BoundingBox(x1=0.2 * 447, x2=0.85 * 447, y1=0.3 * 298, y2=0.95 * 298),
            BoundingBox(x1=0.4 * 447, x2=0.65 * 447, y1=0.1 * 298, y2=0.4 * 298)
        ], shape=image.shape)

        image_aug, bbs_aug = imgaug_augmenters.Affine(rotate=45)(image=image, bounding_boxes=bbs)
        imgaug.imshow(bbs_aug.draw_on_image(image_aug))
        pass

    @staticmethod
    def demo_21_bounding_box_rotate_highlight():
        image = LoadAndAugmentAnImage.demo_1_read_image(
            result_file="./image/Yellow-headed_caracara_Milvago_chimachima_on_capybara_Hydrochoeris_hydrochaeris.JPG",
            origin_file="https://upload.wikimedia.org/wikipedia/commons/8/8e/Yellow-headed_caracara_%28Milvago_chimachima%29_on_capybara_%28Hydrochoeris_hydrochaeris%29.JPG")
        image = imgaug.imresize_single_image(image, (298, 447))

        bbs = BoundingBoxesOnImage([
            BoundingBox(x1=0.2 * 447, x2=0.85 * 447, y1=0.3 * 298, y2=0.95 * 298),
            BoundingBox(x1=0.4 * 447, x2=0.65 * 447, y1=0.1 * 298, y2=0.4 * 298)
        ], shape=image.shape)

        # highlight the area of each bounding box
        image_points = np.copy(image)
        colors = [(0, 255, 0), (128, 128, 255)]
        for bb, color in zip(bbs.bounding_boxes, colors):
            image_points[bb.y1_int:bb.y2_int:4, bb.x1_int:bb.x2_int:4] = color
            pass

        # rotate the image with the highlighted bounding box areas
        rot = imgaug_augmenters.Affine(rotate=45)
        image_points_aug, bbs_aug = rot(image=image_points, bounding_boxes=bbs)

        # visualize
        side_by_side = np.hstack([bbs.draw_on_image(image_points, size=2),
                                  bbs_aug.draw_on_image(image_points_aug, size=2)])
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.imshow(side_by_side)

        plt.show()
        pass

    @staticmethod
    def demo_22_bounding_box_draw():
        image = LoadAndAugmentAnImage.demo_1_read_image(
            result_file="./image/Yellow-headed_caracara_Milvago_chimachima_on_capybara_Hydrochoeris_hydrochaeris.JPG",
            origin_file="https://upload.wikimedia.org/wikipedia/commons/8/8e/Yellow-headed_caracara_%28Milvago_chimachima%29_on_capybara_%28Hydrochoeris_hydrochaeris%29.JPG")
        image = imgaug.imresize_single_image(image, (298, 447))

        bbs = BoundingBoxesOnImage([
            BoundingBox(x1=0.2 * 447, x2=0.85 * 447, y1=0.3 * 298, y2=0.95 * 298),
            BoundingBox(x1=0.4 * 447, x2=0.65 * 447, y1=0.1 * 298, y2=0.4 * 298)
        ], shape=image.shape)

        image_bbs = np.copy(image)
        image_bbs = bbs.bounding_boxes[0].draw_on_image(image_bbs, color=[255, 0, 0], size=3)
        image_bbs = bbs.bounding_boxes[1].draw_on_image(image_bbs, color=[0, 255, 0], size=10, alpha=0.5)
        imgaug.imshow(image_bbs)
        pass

    @staticmethod
    def demo_23_bounding_box_extract():
        image = LoadAndAugmentAnImage.demo_1_read_image(
            result_file="./image/Yellow-headed_caracara_Milvago_chimachima_on_capybara_Hydrochoeris_hydrochaeris.JPG",
            origin_file="https://upload.wikimedia.org/wikipedia/commons/8/8e/Yellow-headed_caracara_%28Milvago_chimachima%29_on_capybara_%28Hydrochoeris_hydrochaeris%29.JPG")
        image = imgaug.imresize_single_image(image, (298, 447))

        bbs = BoundingBoxesOnImage([
            BoundingBox(x1=0.2 * 447, x2=0.85 * 447, y1=0.3 * 298, y2=0.95 * 298),
            BoundingBox(x1=0.4 * 447, x2=0.65 * 447, y1=0.1 * 298, y2=0.4 * 298)
        ], shape=image.shape)

        bird = bbs.bounding_boxes[1].extract_from_image(image)
        imgaug.imshow(bird)

        bird = bbs.bounding_boxes[1].extend(all_sides=10, left=100).extract_from_image(image)
        imgaug.imshow(bird)

        bb = bbs.bounding_boxes[1].shift(left=200)
        imgaug.imshow(bb.draw_on_image(image, size=2))
        imgaug.imshow(bb.extract_from_image(image))
        pass

    @staticmethod
    def demo_24_bounding_box_clip():
        image = LoadAndAugmentAnImage.demo_1_read_image(
            result_file="./image/Yellow-headed_caracara_Milvago_chimachima_on_capybara_Hydrochoeris_hydrochaeris.JPG",
            origin_file="https://upload.wikimedia.org/wikipedia/commons/8/8e/Yellow-headed_caracara_%28Milvago_chimachima%29_on_capybara_%28Hydrochoeris_hydrochaeris%29.JPG")
        image = imgaug.imresize_single_image(image, (298, 447))

        bbs = BoundingBoxesOnImage([
            BoundingBox(x1=0.2 * 447, x2=0.85 * 447, y1=0.3 * 298, y2=0.95 * 298),
            BoundingBox(x1=0.4 * 447, x2=0.65 * 447, y1=0.1 * 298, y2=0.4 * 298)
        ], shape=image.shape)

        Tools.print("----------------")
        Tools.print("Shifted by 200px")
        Tools.print("----------------")
        bb = bbs.bounding_boxes[1].shift(left=200).clip_out_of_image(image.shape)
        if bb.area > 0:
            Tools.print("BB's area is non-zero")
            imgaug.imshow(bb.draw_on_image(image, thickness=2))
        else:
            Tools.print("BB's area is zero: BB: %s vs. image shape: %s" % (bb, image.shape))
            pass

        Tools.print("----------------")
        Tools.print("Shifted by 400px")
        Tools.print("----------------")
        bb = bbs.bounding_boxes[1].shift(left=400).clip_out_of_image(image.shape)
        if bb.area > 0:
            Tools.print("BB's area is non-zero")
            imgaug.imshow(bb.draw_on_image(image, thickness=2))
        else:
            Tools.print("BB's area is zero. BB: %s vs. image shape: %s" % (bb, image.shape))
            pass

        pass

    @staticmethod
    def demo_25_bounding_box_intersection_union_iou():
        image = LoadAndAugmentAnImage.demo_1_read_image(
            result_file="./image/Yellow-headed_caracara_Milvago_chimachima_on_capybara_Hydrochoeris_hydrochaeris.JPG",
            origin_file="https://upload.wikimedia.org/wikipedia/commons/8/8e/Yellow-headed_caracara_%28Milvago_chimachima%29_on_capybara_%28Hydrochoeris_hydrochaeris%29.JPG")
        image = imgaug.imresize_single_image(image, (298, 447))

        bbs = BoundingBoxesOnImage([
            BoundingBox(x1=0.2 * 447, x2=0.85 * 447, y1=0.3 * 298, y2=0.95 * 298),
            BoundingBox(x1=0.4 * 447, x2=0.65 * 447, y1=0.1 * 298, y2=0.4 * 298)
        ], shape=image.shape)

        #####################################################################################################
        # intersection
        bb_intersection = bbs.bounding_boxes[0].intersection(bbs.bounding_boxes[1])
        imgaug.imshow(bb_intersection.draw_on_image(image))
        Tools.print("The intersection has a height of %.4f, width of %.4f and an area of %.4f" % (
            bb_intersection.height, bb_intersection.width, bb_intersection.area))
        #####################################################################################################
        # union
        bb_union = bbs.bounding_boxes[0].union(bbs.bounding_boxes[1])
        imgaug.imshow(bb_union.draw_on_image(image, thickness=2))
        Tools.print("The union has a height of %.4f, width of %.4f and an area of %.4f." % (
            bb_union.height, bb_union.width, bb_union.area))
        #####################################################################################################
        # iou
        # Shift one BB down so that the BBs overlap
        bbs_shifted = imgaug.BoundingBoxesOnImage([bbs.bounding_boxes[0],
                                                   bbs.bounding_boxes[1].shift(top=100)], shape=bbs.shape)
        # Compute IoU without shift
        iou = bbs.bounding_boxes[0].iou(bbs.bounding_boxes[1])
        Tools.print("The IoU of the bounding boxes is: %.4f." % (iou,))
        # Compute IoU after shift
        iou_shifted = bbs.bounding_boxes[0].iou(bbs_shifted.bounding_boxes[1])
        Tools.print("The IoU of the bounding boxes after shifting one box is: %.4f." % (iou_shifted,))
        # Visualize unshifted and shifted BBs
        imgaug.imshow(np.hstack([bbs.draw_on_image(image, size=3), bbs_shifted.draw_on_image(image, size=3)]))
        #####################################################################################################
        pass

    @staticmethod
    def demo_26_bounding_box_project():
        image = LoadAndAugmentAnImage.demo_1_read_image(
            result_file="./image/Yellow-headed_caracara_Milvago_chimachima_on_capybara_Hydrochoeris_hydrochaeris.JPG",
            origin_file="https://upload.wikimedia.org/wikipedia/commons/8/8e/Yellow-headed_caracara_%28Milvago_chimachima%29_on_capybara_%28Hydrochoeris_hydrochaeris%29.JPG")
        image = imgaug.imresize_single_image(image, (298, 447))

        bbs = BoundingBoxesOnImage([
            BoundingBox(x1=0.2 * 447, x2=0.85 * 447, y1=0.3 * 298, y2=0.95 * 298),
            BoundingBox(x1=0.4 * 447, x2=0.65 * 447, y1=0.1 * 298, y2=0.4 * 298)
        ], shape=image.shape)

        # we limit the example here to the bounding box of the bird
        bb_bird = bbs.bounding_boxes[1]
        bbsoi_bird = imgaug.BoundingBoxesOnImage([bbs.bounding_boxes[1]], shape=image.shape)

        # lets resize the original image to twice its width
        image_larger = imgaug.imresize_single_image(image, (1.0, 2.0))

        # we draw what would happen without any change to the bounding box
        Tools.print("BoundingBox without changes:")
        imgaug.imshow(bb_bird.draw_on_image(image_larger, size=3))

        # now the change it using project()
        Tools.print("BoundingBox with project(from, to):")
        imgaug.imshow(bb_bird.project(from_shape=image.shape,
                                      to_shape=image_larger.shape).draw_on_image(image_larger, size=3))

        # and now we do the same two steps for BoundingBoxesOnImage, though here the method is called .on()
        Tools.print("BoundingBoxesOnImage without changes:")
        imgaug.imshow(bbsoi_bird.draw_on_image(image_larger, size=3))

        Tools.print("BoundingBoxesOnImage with on(shape):")
        imgaug.imshow(bbsoi_bird.on(image_larger.shape).draw_on_image(image_larger, size=3))
        pass

    pass


# B3
class AugmentPolygons(object):

    ##################################################################################################
    # https://nbviewer.jupyter.org/github/aleju/imgaug-doc/blob/master/notebooks/B03%20-%20Augment%20Polygons.ipynb
    # Augment Polygons
    ##################################################################################################

    # left meerkat
    meerkat_left = Polygon([
        (350, 100),  # top left
        (390, 85),  # top
        (435, 110),  # top right
        (435, 170),
        (445, 250),  # right elbow
        (430, 290),  # right hip
        (440, 300),
        (420, 340),
        (440, 400),
        (410, 450),  # right foot
        (320, 430),
        (280, 410),  # left foot
        (300, 350),
        (300, 210),  # left elbow
        (340, 160),
        (325, 140)  # nose
    ])

    # center meerkat
    meerkat_center = Polygon([
        (430, 120),  # left top (nose)
        (510, 90),  # top
        (550, 95),  # top right
        (570, 120),  # ear
        (600, 230),
        (600, 450),
        (560, 510),  # bottom right
        (450, 480),  # bottom left
        (430, 340),
        (450, 250),  # elbow
        (500, 165),  # neck
        (430, 145)
    ])

    # right meerkat
    meerkat_right = Polygon([
        (610, 95),  # nose top
        (650, 60),  # top
        (680, 50),  # top
        (710, 60),
        (730, 80),  # top right
        (730, 140),
        (755, 220),
        (750, 340),
        (730, 380),
        (740, 420),
        (715, 560),  # right foot
        (690, 550),
        (680, 470),
        (640, 530),
        (590, 500),  # left foot
        (605, 240),  # left elbow
        (655, 130),  # neck
        (620, 120),  # mouth, bottom
    ])

    image = LoadAndAugmentAnImage.demo_1_read_image(
        result_file="./image/Meerkat_Suricata_suricatta_Tswalu.jpg",
        origin_file="https://upload.wikimedia.org/wikipedia/commons/9/9a/Meerkat_%28Suricata_suricatta%29_Tswalu.jpg")
    image = imgaug.imresize_single_image(image, 0.25)

    @classmethod
    def demo_polygons(cls):
        # 1
        image = np.copy(cls.image)
        meerkat_left = cls.meerkat_left
        meerkat_center = cls.meerkat_center
        meerkat_right = cls.meerkat_right

        # 2
        image_polys = np.copy(image)
        image_polys = meerkat_left.draw_on_image(image_polys, alpha_face=0.2, size_points=7)
        image_polys = meerkat_center.draw_on_image(image_polys, alpha_face=0.2, size_points=7)
        image_polys = meerkat_right.draw_on_image(image_polys, alpha_face=0.2, size_points=7)

        # 3
        psoi_polys = np.copy(image)
        psoi = imgaug.PolygonsOnImage([meerkat_left, meerkat_center, meerkat_right], shape=image.shape)
        psoi_polys = psoi.draw_on_image(psoi_polys, size=5)

        imgaug.imshow(np.hstack([image, image_polys, psoi_polys]))
        pass

    @classmethod
    def demo_polygons_augmenter(cls):
        image = np.copy(cls.image)
        meerkat_left = cls.meerkat_left
        meerkat_center = cls.meerkat_center
        meerkat_right = cls.meerkat_right

        # 1
        psoi = imgaug.PolygonsOnImage([meerkat_left, meerkat_center, meerkat_right], shape=image.shape)

        # 2
        aug = imgaug_augmenters.Sequential([imgaug_augmenters.AdditiveGaussianNoise(scale=10),
                                            imgaug_augmenters.CoarseDropout(0.1, size_px=8),
                                            imgaug_augmenters.AddToHueAndSaturation((-50, 50))])
        image_aug, psoi_aug = aug(image=image, polygons=psoi)
        imgaug.imshow(psoi_aug.draw_on_image(image_aug, alpha_face=0.2, size_points=7))

        # 3
        aug = imgaug_augmenters.Sequential([imgaug_augmenters.Affine(translate_percent={"x": 0.2, "y": 0.1}),
                                            imgaug_augmenters.Fliplr(1.0)])
        image_aug, psoi_aug = aug(image=image, polygons=psoi)
        imgaug.imshow(psoi_aug.draw_on_image(image_aug, alpha_face=0.2, size_points=7))
        pass

    @classmethod
    def demo_polygons_transforms_polygons_bounding_boxes(cls):
        """
        transforms the polygons to bounding boxes,
        then augments image, bounding boxes and polygons and visualizes the results.
        :return:
        """

        image = np.copy(cls.image)
        meerkat_left = cls.meerkat_left
        meerkat_center = cls.meerkat_center
        meerkat_right = cls.meerkat_right

        # 1
        psoi = imgaug.PolygonsOnImage([meerkat_left, meerkat_center, meerkat_right], shape=image.shape)

        # Convert polygons to BBs and put them in BoundingBoxesOnImage instance
        # we will need that instance below to easily draw all augmented BBs on the image
        bbsoi = BoundingBoxesOnImage([polygon.to_bounding_box() for polygon in psoi.polygons], shape=psoi.shape)

        # augment image, BBs and polygons
        batch_aug = imgaug_augmenters.Affine(rotate=45)(images=[image], bounding_boxes=bbsoi,
                                                        polygons=psoi, return_batch=True)

        images_aug = batch_aug.images_aug
        bbsoi_aug = batch_aug.bounding_boxes_aug
        psoi_aug = batch_aug.polygons_aug

        # visualize
        imgaug.imshow(psoi_aug.draw_on_image(bbsoi_aug.draw_on_image(images_aug[0], size=3),
                                             alpha_face=0.2, size_points=7))
        pass

    @classmethod
    def demo_many_consecutive_augmentations(cls):
        image = np.copy(cls.image)
        meerkat_left = cls.meerkat_left
        meerkat_center = cls.meerkat_center
        meerkat_right = cls.meerkat_right

        # 1
        psoi = imgaug.PolygonsOnImage([meerkat_left, meerkat_center, meerkat_right], shape=image.shape)

        # 2
        aug = imgaug_augmenters.Sequential([
            imgaug_augmenters.Affine(rotate=(-20, 20), translate_percent=(-0.2, 0.2),
                                     scale=(0.8, 1.2), mode=["constant", "edge"], cval=0),
            imgaug_augmenters.Fliplr(0.5),
            imgaug_augmenters.PerspectiveTransform((0.01, 0.1)),
            imgaug_augmenters.AddToHueAndSaturation((-20, 20)),
            imgaug_augmenters.LinearContrast((0.8, 1.2), per_channel=0.5),
            imgaug_augmenters.Sometimes(0.75, imgaug_augmenters.Snowflakes())
        ])

        # 3
        images_polys_aug = []
        for _ in range(2 * 4):
            image_aug, psoi_aug = aug(image=image, polygons=psoi)
            image_polys_aug = psoi_aug.draw_on_image(image_aug, alpha_face=0.2, size_points=11)
            images_polys_aug.append(imgaug.imresize_single_image(image_polys_aug, 0.5))
            pass

        imgaug.imshow(imgaug.draw_grid(images_polys_aug, cols=2))
        pass

    @classmethod
    def demo_drawing_polygons(cls):
        image = np.copy(cls.image)
        meerkat_left = cls.meerkat_left
        meerkat_center = cls.meerkat_center
        meerkat_right = cls.meerkat_right

        # 1
        images_polys_aug = [image]
        psoi = imgaug.PolygonsOnImage([meerkat_left, meerkat_center, meerkat_right], shape=image.shape)

        # 2
        image = np.copy(cls.image)
        images_polys_aug.append(psoi.draw_on_image(image))

        # 3
        image_polys = np.copy(image)
        image_polys = meerkat_left.draw_on_image(image_polys, color=[255, 0, 0])
        image_polys = meerkat_center.draw_on_image(image_polys, color=[0, 0, 255])
        image_polys = meerkat_right.draw_on_image(image_polys, color=[128, 64, 128])
        images_polys_aug.append(image_polys)

        # 4
        image_polys = np.copy(image)
        image_polys = meerkat_left.draw_on_image(image_polys, color=[255, 0, 0],
                                                 color_lines=[255, 255, 255], size_points=7)
        image_polys = meerkat_center.draw_on_image(image_polys, color_face=[0, 0, 255],
                                                   color_lines=[255, 0, 0], size_points=7)
        image_polys = meerkat_right.draw_on_image(image_polys, color_points=[255, 0, 0], size_points=7)
        images_polys_aug.append(image_polys)

        # 5
        image_polys = np.copy(image)
        image_polys = meerkat_left.draw_on_image(image_polys, alpha=0.2, size_points=11)
        image_polys = meerkat_center.draw_on_image(image_polys, alpha=0.1, alpha_lines=0.5,
                                                   alpha_face=0.2, alpha_points=1.0, size_points=11)
        image_polys = meerkat_right.draw_on_image(image_polys, color=[0, 0, 255], alpha_face=0, alpha_points=0)
        images_polys_aug.append(image_polys)

        # 6
        image_polys = np.copy(image)
        image_polys = meerkat_left.draw_on_image(image_polys, alpha_face=0.1, size=3)
        image_polys = meerkat_center.draw_on_image(image_polys, alpha_face=0.1, size_lines=7, size_points=3)
        image_polys = meerkat_right.draw_on_image(image_polys, alpha_face=0.1, size_lines=1, size_points=7)
        images_polys_aug.append(image_polys)

        imgaug.imshow(imgaug.draw_grid(images_polys_aug, cols=2))
        pass

    @classmethod
    def demo_extracting_image_content(cls):
        image = np.copy(cls.image)
        meerkat_left = cls.meerkat_left
        meerkat_center = cls.meerkat_center
        meerkat_right = cls.meerkat_right

        images_polys_aug = [image, meerkat_left.extract_from_image(image),
                            meerkat_center.extract_from_image(image),
                            meerkat_right.extract_from_image(image)]
        imgaug.imshow(imgaug.draw_grid(images_polys_aug, cols=4))
        pass

    @classmethod
    def demo_clipping_polygons(cls):
        image = np.copy(cls.image)
        meerkat_left = cls.meerkat_left
        meerkat_center = cls.meerkat_center
        meerkat_right = cls.meerkat_right

        # 1
        images_polys_aug = [image]
        psoi = imgaug.PolygonsOnImage([meerkat_left, meerkat_center, meerkat_right], shape=image.shape)

        # 2
        image_2 = np.copy(image)
        image_aug, psoi_aug = imgaug_augmenters.Affine(translate_px={"y": 200, "x": 300})(image=image_2, polygons=psoi)
        images_polys_aug.append(psoi_aug.draw_on_image(image_aug))

        # 3
        image_3 = np.copy(image)
        image_aug, psoi_aug = imgaug_augmenters.Affine(translate_px={"y": 200, "x": 300})(image=image_3, polygons=psoi)
        image_aug_pad = imgaug.pad(image_aug, bottom=200, right=300)
        images_polys_aug.append(psoi_aug.draw_on_image(image_aug_pad))

        # 4 ???
        image_4 = np.copy(image)
        image_aug, psoi_aug = imgaug_augmenters.Affine(translate_px={"y": 200, "x": 300})(image=image_4, polygons=psoi)
        image_aug_pad = imgaug.pad(image_aug, bottom=200, right=300)
        psoi_aug_removed = psoi_aug.remove_out_of_image(fully=True, partly=False)
        images_polys_aug.append(psoi_aug_removed.draw_on_image(image_aug_pad))

        # 5
        image_5 = np.copy(image)
        image_aug, psoi_aug = imgaug_augmenters.Affine(translate_px={"y": 200, "x": 300})(image=image_5, polygons=psoi)
        image_aug_pad = imgaug.pad(image_aug, bottom=200, right=300)
        psoi_aug_removed = psoi_aug.remove_out_of_image(fully=True, partly=False)
        psoi_aug_removed_clipped = psoi_aug_removed.clip_out_of_image()
        images_polys_aug.append(psoi_aug_removed_clipped.draw_on_image(image_aug_pad))

        imgaug.imshow(imgaug.draw_grid(images_polys_aug, cols=2))
        pass

    @classmethod
    def demo_computing_height_width_area(cls):
        image = np.copy(cls.image)
        meerkat_left = cls.meerkat_left
        meerkat_center = cls.meerkat_center
        meerkat_right = cls.meerkat_right

        Tools.print("Heights: %.2f, %.2f, %.2f" % (meerkat_left.height, meerkat_center.height, meerkat_right.height))
        Tools.print("Widths : %.2f, %.2f, %.2f" % (meerkat_left.width, meerkat_center.width, meerkat_right.width))
        Tools.print("Areas  : %.2f, %.2f, %.2f" % (meerkat_left.area, meerkat_center.area, meerkat_right.area))
        pass

    @classmethod
    def demo_modifying_polygon_start_point(cls):
        image = np.copy(cls.image)
        meerkat_left = cls.meerkat_left
        meerkat_center = cls.meerkat_center
        meerkat_right = cls.meerkat_right

        # 1
        images_polys_aug = [image, image]

        # 2
        image_poly = meerkat_left.draw_on_image(image, size_points=11, alpha_face=0, size=5)
        images_polys_aug.append(image_poly)
        first_point = meerkat_left.exterior[0]
        image_poly = Keypoint(x=first_point[0], y=first_point[1]).draw_on_image(image_poly, size=15)
        images_polys_aug.append(image_poly)

        # 3
        meerkat_left_reordered = meerkat_left.change_first_point_by_index(2)
        image_poly = meerkat_left_reordered.draw_on_image(image, size_points=11, alpha_face=0, size=5)
        images_polys_aug.append(image_poly)
        first_point = meerkat_left_reordered.to_keypoints()[0]
        image_poly = first_point.draw_on_image(image_poly, size=15)
        images_polys_aug.append(image_poly)

        # 4
        meerkat_left_reordered = meerkat_left.change_first_point_by_coords(y=image.shape[0],
                                                                           x=image.shape[1], max_distance=None)
        image_poly = meerkat_left_reordered.draw_on_image(image, size_points=11, alpha_face=0, size=5)
        images_polys_aug.append(image_poly)
        first_point = meerkat_left_reordered.to_keypoints()[0]
        image_poly = first_point.draw_on_image(image_poly, size=15)
        images_polys_aug.append(image_poly)

        imgaug.imshow(imgaug.draw_grid(images_polys_aug, cols=2))
        pass

    @classmethod
    def demo_converting_to_bounding_boxes(cls):
        image = np.copy(cls.image)
        meerkat_left = cls.meerkat_left
        meerkat_center = cls.meerkat_center
        meerkat_right = cls.meerkat_right

        # 1
        images_polys_aug = [image, image]
        psoi = imgaug.PolygonsOnImage([meerkat_left, meerkat_center, meerkat_right], shape=image.shape)

        # 2
        meerkat_left_bb = meerkat_left.to_bounding_box()
        meerkat_center_bb = meerkat_center.to_bounding_box()
        meerkat_right_bb = meerkat_right.to_bounding_box()

        # 3
        image_bbs = psoi.draw_on_image(image, alpha=1.0, alpha_face=0, size_points=11)
        images_polys_aug.append(image_bbs)
        image_bbs = meerkat_left_bb.draw_on_image(image_bbs, size=5, color=[255, 0, 0])
        images_polys_aug.append(image_bbs)
        image_bbs = meerkat_center_bb.draw_on_image(image_bbs, size=5, color=[0, 0, 255])
        images_polys_aug.append(image_bbs)
        image_bbs = meerkat_right_bb.draw_on_image(image_bbs, size=5, color=[255, 255, 255])
        images_polys_aug.append(image_bbs)

        imgaug.imshow(imgaug.draw_grid(images_polys_aug, cols=2))
        pass

    @classmethod
    def demo_comparing_polygons(cls):
        meerkat_left = cls.meerkat_left
        meerkat_center = cls.meerkat_center
        meerkat_right = cls.meerkat_right

        Tools.print(meerkat_left.exterior_almost_equals(meerkat_left))
        Tools.print(meerkat_left.exterior_almost_equals(meerkat_right))

        meerkat_left_shifted = meerkat_left.shift(left=1)
        Tools.print(meerkat_left.exterior_almost_equals(meerkat_left_shifted))
        Tools.print(meerkat_left.exterior_almost_equals(meerkat_left_shifted, max_distance=1))
        Tools.print(meerkat_left.exterior_almost_equals(meerkat_left_shifted, max_distance=1.01))
        pass

    pass


# B4
class AugmentHeatmaps(object):

    ##################################################################################################
    # https://nbviewer.jupyter.org/github/aleju/imgaug-doc/blob/master/notebooks/B04%20-%20Augment%20Heatmaps.ipynb
    # Augment Heatmaps
    ##################################################################################################

    @classmethod
    def demo_loading_example_data(cls):
        image = imgaug.quokka(size=0.25)
        heatmap = imgaug.quokka_heatmap(size=0.25)

        imgaug.imshow(np.hstack([image, heatmap.draw()[0]]))

        Tools.print(type(heatmap))
        Tools.print(type(heatmap.arr_0to1))  # the numpy heatmap array, normalized to [0.0, 1.0]
        Tools.print("{} {}".format(np.min(heatmap.arr_0to1), np.max(heatmap.arr_0to1)))
        Tools.print("{} {}".format(image.shape, heatmap.arr_0to1.shape))
        pass

    @classmethod
    def demo_augmenting_example_heatmap(cls):
        image = imgaug.quokka(size=0.25)
        heatmap = imgaug.quokka_heatmap(size=0.25)

        seq = imgaug_augmenters.Sequential([imgaug_augmenters.Dropout(0.2),
                                            imgaug_augmenters.Affine(rotate=(-45, 45))])
        image_aug, heatmap_aug = seq(image=image, heatmaps=heatmap)

        imgaug.imshow(np.hstack([image_aug, heatmap_aug.draw()[0], heatmap_aug.draw_on_image(image_aug)[0]]))
        pass

    @classmethod
    def demo_augment_heatmap_with_lower_resolution(cls):
        image = imgaug.quokka(size=0.25)
        Tools.print("Image size: {}".format(image.shape))

        heatmap = imgaug.quokka_heatmap(size=0.25)
        Tools.print("Original heatmap size: {}".format(heatmap.arr_0to1.shape))

        heatmap_smaller = heatmap.resize(0.25)
        Tools.print("Resized heatmap size: {}".format(heatmap_smaller.arr_0to1.shape))
        imgaug.imshow(heatmap_smaller.draw()[0])

        seq = imgaug_augmenters.Sequential([imgaug_augmenters.Dropout(0.2),
                                            imgaug_augmenters.Affine(rotate=(-45, 45))])
        image_aug, heatmap_smaller_aug = seq(image=image, heatmaps=heatmap_smaller)

        imgaug.imshow(np.hstack([image_aug, heatmap_smaller_aug.draw(size=image_aug.shape[0:2])[0],
                                 heatmap_smaller_aug.draw_on_image(image_aug)[0]]))
        pass

    @classmethod
    def demo_create_heatmap_from_numpy_array(cls):
        image = imgaug.quokka(size=0.25)

        arr = np.tile(np.linspace(0, 1.0, num=128).astype(np.float32).reshape((1, 128)), (128, 1))  # (128, 128)
        heatmap = HeatmapsOnImage(arr, shape=image.shape)
        imgaug.imshow(heatmap.draw_on_image(image)[0])

        # first channel, horizontal gradient
        arr0 =  np.tile(np.linspace(0, 1.0, num=128).astype(np.float32).reshape((1, 128)), (128, 1))  # (128, 128)
        # second channel, set horizontal subarea to low value
        arr1 = np.copy(arr0)
        arr1[30:-30, :] *= 0.25
        # third channel, set vertical subarea to low value
        arr2 = np.copy(arr0)
        arr2[:, 30:-30] *= 0.25
        # fourth channel, set pixels in regular distances to zero
        arr3 = np.copy(arr0)
        arr3[::4, ::4] = 0

        # create heatmap array and heatmap
        arr = np.dstack([arr0, arr1, arr2, arr3])  # (128, 128, 4)
        heatmaps = HeatmapsOnImage(arr, shape=image.shape)

        # visualize
        heatmaps_drawn = heatmaps.draw_on_image(image)
        imgaug.imshow(np.hstack([heatmaps_drawn[0], heatmaps_drawn[1], heatmaps_drawn[2], heatmaps_drawn[3]]))
        pass

    @classmethod
    def demo_resize_average_pool_and_max_pool_heatmaps(cls):
        image = imgaug.quokka(size=0.25)
        heatmap = imgaug.quokka_heatmap(size=0.25)

        imgaug_list = []
        for factor in [4, 8, 16]:
            resized = heatmap.resize(1 / factor)
            avg_pooled = heatmap.avg_pool(factor)
            max_pooled = heatmap.max_pool(factor)

            # print heatmap sizes after resize/pool
            Tools.print("[shapes] resized: {}, avg pooled: {}, max pooled: {}".format(
                resized.get_arr().shape, avg_pooled.get_arr().shape, max_pooled.get_arr().shape))

            # visualize
            imgaug_list.append(resized.draw_on_image(image)[0])
            imgaug_list.append(avg_pooled.draw_on_image(image)[0])
            imgaug_list.append(max_pooled.draw_on_image(image)[0])
            pass

        imgaug.imshow(imgaug.draw_grid(imgaug_list, cols=3))
        pass

    @classmethod
    def demo_customize_drawing(cls):
        image = imgaug.quokka(size=0.25)
        heatmap = imgaug.quokka_heatmap(size=0.25)
        heatmap_small = heatmap.resize(0.5)

        imgaug_list = [
            # 1
            heatmap.draw()[0],
            heatmap.draw(size=2.0)[0],
            heatmap.draw(size=(50, 500))[0],

            # 2
            heatmap.draw(cmap="gray")[0],
            heatmap.draw(cmap="gnuplot2")[0],
            heatmap.draw(cmap="tab10")[0],

            # 3
            heatmap.draw_on_image(image, cmap="gray")[0],
            heatmap.draw_on_image(image, cmap="gnuplot2")[0],
            heatmap.draw_on_image(image, cmap="tab10")[0],

            # 4
            heatmap.draw_on_image(image, cmap="gnuplot2", alpha=0.25)[0],
            heatmap.draw_on_image(image, cmap="gnuplot2", alpha=0.50)[0],
            heatmap.draw_on_image(image, cmap="gnuplot2", alpha=0.75)[0],

            # 5
            heatmap_small.draw()[0],
            heatmap_small.draw_on_image(image, resize="heatmaps", alpha=0.5)[0],
            heatmap_small.draw_on_image(image, resize="image", alpha=0.5)[0]
        ]

        imgaug.imshow(imgaug.draw_grid(imgaug_list, cols=3))
        pass

    pass


# B5
class AugmentSegmentationMaps(object):

    ##################################################################################################
    # https://nbviewer.jupyter.org/github/aleju/imgaug-doc/blob/master/notebooks/B05%20-%20Augment%20Segmentation%20Maps.ipynb
    # Augment Segmentation Maps
    ##################################################################################################

    image = LoadAndAugmentAnImage.demo_1_read_image(
        result_file="./image/Tamias_striatus_CT.jpg",
        origin_file="https://upload.wikimedia.org/wikipedia/commons/f/f4/Tamias_striatus_CT.jpg")
    image = imgaug.imresize_single_image(image, 0.15)
    Tools.print(image.shape)

    @classmethod
    def _demo_get_segmap(cls):
        image = np.copy(cls.image)

        tree_kps_xy = np.float32([[0, 300], [image.shape[1] - 1, 230],
                                  [image.shape[1] - 1, image.shape[0] - 1], [0, image.shape[0] - 1]])
        kpsoi_tree = KeypointsOnImage.from_xy_array(tree_kps_xy, shape=image.shape)
        poly_tree = Polygon(kpsoi_tree.keypoints)

        chipmunk_kps_xy = np.float32([
            [200, 50],  # left ear, top (from camera perspective)
            [220, 70],
            [260, 70],
            [280, 50],  # right ear, top
            [290, 80],
            [285, 110],
            [310, 140],
            [330, 175],  # right of cheek
            [310, 260],  # right of right paw
            [175, 275],  # left of left paw
            [170, 220],
            [150, 200],
            [150, 170],  # left of cheek
            [160, 150],
            [186, 120],  # left of eye
            [185, 70]
        ])
        kpsoi_chipmunk = KeypointsOnImage.from_xy_array(chipmunk_kps_xy, shape=image.shape)
        poly_chipmunk = Polygon(kpsoi_chipmunk.keypoints)

        segmap = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        segmap = poly_tree.draw_on_image(segmap, color=(0, 255, 0), alpha=1.0, alpha_lines=0.0, alpha_points=0.0)
        segmap = poly_chipmunk.draw_on_image(segmap, color=(0, 0, 255), alpha=1.0, alpha_lines=0.0, alpha_points=0.0)
        segmap = np.asarray(np.argmax(segmap, axis=2), dtype=np.int32)
        segmap = SegmentationMapOnImage(segmap, nb_classes=3, shape=image.shape)
        return segmap

    @classmethod
    def demo_loading_example_data(cls):
        image = np.copy(cls.image)

        # 1
        imgaug_list = [image]

        tree_kps_xy = np.float32([[0, 300], [image.shape[1] - 1, 230],
                                  [image.shape[1] - 1, image.shape[0] - 1], [0, image.shape[0] - 1]])

        # 2
        image = np.copy(cls.image)
        kpsoi_tree = KeypointsOnImage.from_xy_array(tree_kps_xy, shape=image.shape)
        imgaug_list.append(kpsoi_tree.draw_on_image(image, size=13))

        chipmunk_kps_xy = np.float32([
            [200, 50],  # left ear, top (from camera perspective)
            [220, 70],
            [260, 70],
            [280, 50],  # right ear, top
            [290, 80],
            [285, 110],
            [310, 140],
            [330, 175],  # right of cheek
            [310, 260],  # right of right paw
            [175, 275],  # left of left paw
            [170, 220],
            [150, 200],
            [150, 170],  # left of cheek
            [160, 150],
            [186, 120],  # left of eye
            [185, 70]
        ])

        # 3
        image = np.copy(cls.image)
        kpsoi_chipmunk = KeypointsOnImage.from_xy_array(chipmunk_kps_xy, shape=image.shape)
        imgaug_list.append(kpsoi_chipmunk.draw_on_image(image, size=7))

        # 4
        image = np.copy(cls.image)
        poly_tree = Polygon(kpsoi_tree.keypoints)
        poly_chipmunk = Polygon(kpsoi_chipmunk.keypoints)
        imgaug_list.append(poly_tree.draw_on_image(image))
        imgaug_list.append(poly_chipmunk.draw_on_image(image))

        # 5
        segmap = cls._demo_get_segmap()
        imgaug_list.append(segmap.draw_on_image(image)[0])

        imgaug.imshow(imgaug.draw_grid(imgaug_list, cols=2))
        pass

    @classmethod
    def demo_augment_example_segmentation_map(cls):
        image = np.copy(cls.image)
        segmap = cls._demo_get_segmap()

        # 1
        imgaug_list = [image]

        # 2
        seq = imgaug_augmenters.Sequential([
            imgaug_augmenters.CoarseDropout(0.1, size_percent=0.2),
            imgaug_augmenters.Affine(rotate=(-30, 30)),
            imgaug_augmenters.ElasticTransformation(alpha=10, sigma=1)
        ])
        image_aug, segmap_aug = seq(image=image, segmentation_maps=segmap)
        imgaug_list.append(segmap_aug.draw_on_image(image_aug)[0])
        imgaug_list.append(segmap_aug.draw()[0])

        # 3
        aug = imgaug_augmenters.ElasticTransformation(alpha=200, sigma=10)
        image_aug, segmap_aug = aug(image=image, segmentation_maps=segmap)
        image_aug_unaligned = aug(image=image)
        segmap_aug_unaligned = aug(segmentation_maps=segmap)
        imgaug_list.append(segmap_aug.draw_on_image(image_aug, alpha=0.4)[0])
        imgaug_list.append(segmap_aug.draw_on_image(image_aug_unaligned, alpha=0.4)[0])
        imgaug_list.append(segmap_aug_unaligned.draw_on_image(image_aug_unaligned, alpha=0.4)[0])

        imgaug.imshow(imgaug.draw_grid(imgaug_list, cols=2))
        pass

    @classmethod
    def demo_scaling_segmentation_maps(cls):
        image = np.copy(cls.image)
        segmap = cls._demo_get_segmap()

        imgaug_list = [image]

        segmap_small = segmap.resize(0.25)
        Tools.print("Before: {}, After: {}".format(segmap.arr.shape, segmap_small.arr.shape))

        imgaug_list.append(segmap_small.draw()[0])
        imgaug_list.append(segmap.draw_on_image(image)[0])
        imgaug_list.append(segmap_small.draw_on_image(image)[0])

        imgaug.imshow(imgaug.draw_grid(imgaug_list, cols=2))
        pass

    @classmethod
    def demo_augment_segmentation_maps_smaller_than_their_corresponding_image(cls):
        image = np.copy(cls.image)
        segmap = cls._demo_get_segmap()
        segmap_small = segmap.resize(0.25)

        Tools.print("Image size: %s Segmentation Map size: %s (on image with shape: %s)" % (
            image.shape, segmap_small.arr.shape, segmap_small.shape))

        imgaug_list = [image]

        aug = imgaug_augmenters.ElasticTransformation(alpha=200, sigma=10)
        image_aug, segmap_small_aug = aug(image=image, segmentation_maps=segmap_small)

        imgaug_list.append(segmap_small_aug.draw_on_image(image_aug, alpha=0.4)[0])

        aug = imgaug_augmenters.Crop(px=(0, 0, 50, 200))  # (top, right, bottom, left)
        aug_det = aug.to_deterministic()
        image_aug = aug_det.augment_image(image)  # augment image
        segmap_aug = aug_det.augment_segmentation_maps(segmap)  # augment normal-sized segmentation map
        segmap_small_aug = aug_det.augment_segmentation_maps(segmap_small)  # augment smaller-sized segmentation map

        imgaug_list.append(segmap_aug.draw_on_image(image_aug, alpha=0.4)[0])
        imgaug_list.append(segmap_small_aug.draw_on_image(image_aug, alpha=0.4)[0])

        imgaug.imshow(imgaug.draw_grid(imgaug_list, cols=2))
        pass

    @classmethod
    def demo_draw_segmentation_maps(cls):
        image = np.copy(cls.image)
        segmap = cls._demo_get_segmap()

        segmap_aug = imgaug_augmenters.ElasticTransformation(alpha=50, sigma=5).augment_segmentation_maps(segmap)

        image_seg, foreground_mask = segmap.draw(return_foreground_mask=True)  # ???????????????????????
        image_seg_masked = np.copy(image_seg)
        image_seg_masked[foreground_mask] = [255, 255, 255]

        image_seg_2, foreground_mask = segmap.draw(background_class_id=2, return_foreground_mask=True)
        image_seg_masked_2 = np.copy(image_seg)
        image_seg_masked_2[foreground_mask] = [255, 255, 255]
        image_seg_masked_2[~foreground_mask] = [0, 0, 0]

        segmap_small = segmap.resize(0.2)
        print(segmap_small.arr.shape, image.shape)

        imgaug_list = [
            # 1
            image,
            image,
            segmap.draw()[0],
            segmap_aug.draw()[0],

            image_seg,
            image_seg_masked,
            image_seg_2,
            image_seg_masked_2,

            image,
            segmap.draw(colors=[(255, 255, 255), (128, 128, 0), (0, 0, 128)])[0],
            segmap.draw(0.1),
            segmap.draw_on_image(image),

            image,
            segmap.draw_on_image(image, alpha=0.25),
            segmap.draw_on_image(image, alpha=0.50),
            segmap.draw_on_image(image, alpha=0.75),

            segmap.draw_on_image(image, draw_background=True),
            segmap.draw_on_image(image, draw_background=False, background_class_id=2),
            segmap_small.draw_on_image(image),
            segmap_small.draw_on_image(image, resize="image")
        ]

        imgaug.imshow(imgaug.draw_grid(imgaug_list, cols=4))
        pass

    @classmethod
    def demo_change_segmentation_maps_with_non_geometric_augmentations(cls):
        image = np.copy(cls.image)
        segmap = cls._demo_get_segmap()

        imgaug_list = [image]

        aug = imgaug_augmenters.Sequential([imgaug_augmenters.Affine(rotate=(-20, 20)),
                                            imgaug_augmenters.CoarseDropout(0.2, size_percent=0.1),
                                            imgaug_augmenters.AdditiveGaussianNoise(scale=0.2 * 255)])
        image_aug, segmap_aug = aug(image=image, segmentation_maps=segmap)

        imgaug_list.append(segmap_aug.draw_on_image(image_aug)[0])
        imgaug_list.append(segmap_aug.draw()[0])

        aug_images = imgaug_augmenters.Sequential([
            imgaug_augmenters.Affine(rotate=(-20, 20), random_state=1),
            imgaug_augmenters.CoarseDropout(0.2, size_percent=0.05, random_state=2),
            imgaug_augmenters.AdditiveGaussianNoise(scale=0.2 * 255, random_state=3)], random_state=4)
        aug_segmaps = imgaug_augmenters.Sequential([
            imgaug_augmenters.Affine(rotate=(-20, 20), random_state=1),
            imgaug_augmenters.CoarseDropout(0.2, size_percent=0.05, random_state=2)], random_state=4)
        image_aug = aug_images(image=image)
        segmap_arr_aug = aug_segmaps(image=segmap.get_arr_int().astype(np.uint8))

        segmap_aug = imgaug.SegmentationMapOnImage(segmap_arr_aug, nb_classes=3, shape=segmap.shape)

        imgaug_list.append(image_aug)
        imgaug_list.append(segmap_aug.draw_on_image(image_aug, alpha=0.5)[0])
        imgaug_list.append(segmap_aug.draw()[0])

        imgaug.imshow(imgaug.draw_grid(imgaug_list, cols=3))
        pass

    pass


# B6
class AugmentLineStrings(object):
    ##################################################################################################
    # https://nbviewer.jupyter.org/github/aleju/imgaug-doc/blob/master/notebooks/B06%20-%20Augment%20Line%20Strings.ipynb
    # Augment Line Strings
    ##################################################################################################
    pass


# C1: 如何产生随机数
class UsingProbabilityDistributionsAsParameters(object):

    ##################################################################################################
    # https://nbviewer.jupyter.org/github/aleju/imgaug-doc/blob/master/notebooks/C01%20-%20Using%20Probability%20Distributions%20as%20Parameters.ipynb
    # Using Probability Distributions as Parameters
    ##################################################################################################

    @staticmethod
    def demo_creating_and_using_stochastic_parameters():
        np.set_printoptions(precision=2, linewidth=125, suppress=False)

        ##############################################################################################
        # The example below creates an augmenter Multiply and uses once a scalar value, once a tuple and once a list.
        # As the printed messages show, these inputs are automatically converted to stochastic parameters.
        aug = imgaug_augmenters.Multiply(mul=1)
        Tools.print(aug.mul)
        aug = imgaug_augmenters.Multiply(mul=(0.5, 1.5))
        Tools.print(aug.mul)
        aug = imgaug_augmenters.Multiply(mul=[0.5, 1.0, 1.5])
        Tools.print(aug.mul)
        ##############################################################################################
        # Instead of using the automatic conversion, one can also directly provide a stochastic parameter.
        aug = imgaug_augmenters.Multiply(mul=imgaug_parameters.Deterministic(1))
        Tools.print(aug.mul)
        ##############################################################################################
        # continuous probability distributions
        Tools.print(imgaug_parameters.Deterministic(1).draw_samples(10))
        Tools.print(imgaug_parameters.Deterministic(1).draw_samples(10, random_state=np.random.RandomState(1)))
        Tools.print(imgaug_parameters.Deterministic(1).draw_samples(10, random_state=np.random.RandomState(1)))
        Tools.print(imgaug_parameters.Deterministic(1).draw_samples(10, random_state=np.random.RandomState(2)))

        Tools.print(imgaug_parameters.Uniform(0, 1).draw_samples(10))
        Tools.print(imgaug_parameters.Uniform(0, 1).draw_samples(10, random_state=np.random.RandomState(1)))
        Tools.print(imgaug_parameters.Uniform(0, 1).draw_samples(10, random_state=np.random.RandomState(1)))
        Tools.print(imgaug_parameters.Uniform(0, 1).draw_samples(10, random_state=np.random.RandomState(2)))

        Tools.print(imgaug_parameters.Choice([0, 0.5, 1.0]).draw_samples(10))
        Tools.print(imgaug_parameters.Choice([0, 0.5, 1.0]).draw_samples(10, random_state=np.random.RandomState(1)))
        Tools.print(imgaug_parameters.Choice([0, 0.5, 1.0]).draw_samples(10, random_state=np.random.RandomState(1)))
        Tools.print(imgaug_parameters.Choice([0, 0.5, 1.0]).draw_samples(10, random_state=np.random.RandomState(2)))

        Tools.print(imgaug_parameters.Binomial(p=0.5).draw_samples(4 * 4, random_state=np.random.RandomState(1)))
        Tools.print(imgaug_parameters.Binomial(p=0.5).draw_samples((4, 4), random_state=np.random.RandomState(1)))

        Tools.print((imgaug_parameters.Deterministic(1) + 1).draw_samples(10))
        Tools.print((imgaug_parameters.Deterministic(1) - 1).draw_samples(10))
        Tools.print((imgaug_parameters.Deterministic(1) / 2).draw_samples(10))
        Tools.print((imgaug_parameters.Deterministic(1) * 2).draw_samples(10))
        Tools.print((imgaug_parameters.Deterministic(1) ** 2).draw_samples(10))

        Tools.print(imgaug_parameters.Choice([1, 2, 3]))
        Tools.print(imgaug_parameters.Choice([1, 2, 3]) * 2)
        Tools.print(imgaug_parameters.Choice([1, 2, 3]).draw_samples(10))
        Tools.print((imgaug_parameters.Choice([1, 2, 3]) * 2).draw_samples(10))
        ##############################################################################################
        # Discrete Random Variables
        Tools.print(imgaug_parameters.DiscreteUniform(0, 3).draw_samples(20))
        Tools.print(imgaug_parameters.Binomial(p=0.3).draw_samples(20))
        Tools.print(imgaug_parameters.Poisson(lam=1).draw_samples(20))
        ##############################################################################################
        # Any continuous probability distribution can be turned into a discrete one using Discretize,
        # which rounds float values and then converts them to the nearest integers
        Tools.print("Continuous: {}".format(imgaug_parameters.Normal(0.0, 1.0).draw_samples(20)))
        Tools.print("Discrete: {}".format(
            imgaug_parameters.Discretize(imgaug_parameters.Normal(0.0, 1.0)).draw_samples(20)))
        ##############################################################################################
        # In case you have to limit the value range of a probability distribution.
        uniform = imgaug_parameters.Uniform(-2, 2)
        Tools.print(uniform.draw_samples(15))
        Tools.print(imgaug_parameters.Clip(uniform, -1, None).draw_samples(15))
        Tools.print(imgaug_parameters.Clip(uniform, None, 1).draw_samples(15))
        Tools.print(imgaug_parameters.Clip(uniform, -1, 1).draw_samples(15))
        ##############################################################################################
        # uses a histogram to visualize the probability density after applying Clip to Uniform.
        plot = imgaug_parameters.Clip(uniform, -1, 1).draw_distribution_graph()
        plot_large = imgaug.imresize_single_image(plot, 2.0)
        imgaug.imshow(plot_large)
        ##############################################################################################
        # flip signs is simple arithmetic.
        Tools.print(imgaug_parameters.Poisson(lam=1).draw_samples(20))
        Tools.print((-1) * imgaug_parameters.Poisson(lam=1).draw_samples(20))
        ##############################################################################################
        # In case you only want positive values, Absolute can be used, here applied to a gaussian distribution.
        Tools.print(imgaug_parameters.Normal(0, 1.0).draw_samples(15))
        Tools.print(imgaug_parameters.Absolute(imgaug_parameters.Normal(0, 1.0)).draw_samples(15))
        ##############################################################################################
        # To force a specific fraction of random samples to have positive signs.
        # It first turns all values to positive ones and then flips the sign of (1-p_positive) percent.
        rnd_sign = imgaug_parameters.RandomSign(imgaug_parameters.Normal(0, 1.0), p_positive=0.8)
        Tools.print("15 random samples: {}".format(rnd_sign.draw_samples(15)))
        Tools.print("Positive samples in 10000 random samples: {}".format(np.sum(rnd_sign.draw_samples(10000) > 0)))
        ##############################################################################################
        # ForceSign
        force_sign_invert = imgaug_parameters.ForceSign(imgaug_parameters.Choice([-2, -1, 5, 10]),
                                                        positive=False, mode="invert")
        force_sign_reroll = imgaug_parameters.ForceSign(imgaug_parameters.Choice([-2, -1, 5, 10]),
                                                        positive=False, mode="reroll")
        Tools.print(force_sign_invert.draw_samples(15))
        Tools.print(force_sign_reroll.draw_samples(15))

        plot = imgaug_parameters.ForceSign(imgaug_parameters.Normal(4, 2.0),
                                           positive=True, mode="invert").draw_distribution_graph(bins=400)
        plot_large = imgaug.imresize_single_image(plot, 2.0)
        imgaug.imshow(plot_large)

        plot = imgaug_parameters.ForceSign(imgaug_parameters.Normal(4, 2.0),
                                           positive=True, mode="reroll").draw_distribution_graph(bins=400)
        plot_large = imgaug.imresize_single_image(plot, 2.0)
        imgaug.imshow(plot_large)
        ##############################################################################################
        # The helpers Positive() and Negative() are shortcuts for ForceSign(positive=True) and ForceSign(positive=False).
        positive_invert = imgaug_parameters.Positive(imgaug_parameters.Choice([-2, -1, 5, 10]), mode="invert")
        positive_reroll = imgaug_parameters.Positive(imgaug_parameters.Choice([-2, -1, 5, 10]), mode="reroll")
        Tools.print(positive_invert.draw_samples(15))
        Tools.print(positive_reroll.draw_samples(15))
        ##############################################################################################
        # Combining Probability Distributions
        param = imgaug_parameters.Normal(imgaug_parameters.Choice([-2, 2]), 1.0)
        plot = param.draw_distribution_graph()
        plot_large = imgaug.imresize_single_image(plot, 2.0)
        imgaug.imshow(plot_large)

        param = imgaug_parameters.Choice([imgaug_parameters.Normal(-1, 1.0), imgaug_parameters.Normal(1, 0.1)])
        plot = param.draw_distribution_graph()
        plot_large = imgaug.imresize_single_image(plot, 2.0)
        imgaug.imshow(plot_large)
        ##############################################################################################
        pass

    @staticmethod
    def demo_coarse_salt_and_pepper():
        def apply_coarse_salt_and_pepper(image, p, size_px):
            mask = imgaug_parameters.Binomial(p)  # mask where to replace
            # make the mask coarser
            mask_coarse = imgaug_parameters.FromLowerResolution(other_param=mask, size_px=size_px)
            # the noise to use as replacements, mostly close to 0.0 and 1.0
            replacement = imgaug_parameters.Beta(0.5, 0.5) * 255  # project noise to uint8 value range
            # replace masked areas with noise
            return imgaug_augmenters.ReplaceElementwise(mask=mask_coarse, replacement=replacement).augment_image(image)

        image = imgaug.quokka_square(size=(128, 128))  # example images
        image_aug = apply_coarse_salt_and_pepper(image, 0.05, 16)  # apply noise
        imgaug.imshow(imgaug.imresize_single_image(np.hstack([image, image_aug]), 4.0))
        pass

    pass


# C2
class AugSequence:

    def __init__(self):
        # instantiate each augmenter and save it to its own variable
        self.affine = imgaug_augmenters.Affine(rotate=(-20, 20), translate_px={"x": (-10, 10), "y": (-5, 5)})
        self.multiply = imgaug_augmenters.Multiply((0.9, 1.1))
        self.contrast = imgaug_augmenters.LinearContrast((0.8, 1.2))
        self.gray = imgaug_augmenters.Grayscale((0.0, 1.0))
        pass

    def augment_images(self, x):
        # apply each augmenter on its own, one by one
        x = self.affine.augment_images(x)
        x = self.multiply.augment_images(x)
        x = self.contrast.augment_images(x)
        x = self.gray.augment_images(x)
        return x
    pass


# C2： 如何控制Aug
class UsingImgaugWithMoreControlFlow(object):

    ##################################################################################################
    # https://nbviewer.jupyter.org/github/aleju/imgaug-doc/blob/master/notebooks/C02%20-%20Using%20imgaug%20with%20more%20Control%20Flow.ipynb
    # Using imgaug with more Control Flow
    ##################################################################################################

    @staticmethod
    def demo_non_deferred_way():  # 非延迟方式，类似于pytorch。（延迟方式，类似于tensorflow）
        aug = AugSequence()
        image = imgaug.quokka_square(size=(256, 256))
        images_aug = aug.augment_images([image, image])
        imgaug.imshow(np.hstack([image, images_aug[0], images_aug[1]]))
        pass

    @staticmethod
    def demo_single_function():

        def augment_images(x):
            x = imgaug_augmenters.Affine(rotate=(-20, 20)).augment_images(x)
            x = imgaug_augmenters.Multiply((0.9, 1.1)).augment_images(x)
            x = imgaug_augmenters.LinearContrast((0.8, 1.2)).augment_images(x)
            x = imgaug_augmenters.Grayscale((0.0, 1.0)).augment_images(x)
            return x

        image = imgaug.quokka_square(size=(256, 256))
        images_aug = augment_images([image, image])
        imgaug.imshow(np.hstack([image, images_aug[0], images_aug[1]]))
        pass

    @staticmethod
    def demo_single_function_and_different_input_types():

        def augment_images(x, seed):
            x = imgaug_augmenters.Affine(translate_px=(-60, 60), random_state=seed).augment_images(x)
            x = imgaug_augmenters.Multiply((0.9, 1.1), random_state=seed).augment_images(x)
            x = imgaug_augmenters.LinearContrast((0.8, 1.2), random_state=seed).augment_images(x)
            x = imgaug_augmenters.Grayscale((0.0, 1.0), random_state=seed).augment_images(x)
            return x

        def augment_bounding_boxes(x, seed):
            x = imgaug_augmenters.Affine(translate_px=(-60, 60), random_state=seed).augment_bounding_boxes(x)
            x = imgaug_augmenters.Multiply((0.9, 1.1), random_state=seed).augment_bounding_boxes(x)
            x = imgaug_augmenters.LinearContrast((0.8, 1.2), random_state=seed).augment_bounding_boxes(x)
            x = imgaug_augmenters.Grayscale((0.0, 1.0), random_state=seed).augment_bounding_boxes(x)
            return x

        image = imgaug.quokka_square(size=(256, 256))
        images_aug = augment_images([image, image], seed=2)

        bbsoi = imgaug.BoundingBoxesOnImage(
            bounding_boxes=[imgaug.BoundingBox(x1=40, y1=20, x2=230, y2=250)], shape=image.shape)
        bbsois_aug = augment_bounding_boxes([bbsoi, bbsoi], seed=2)

        imgaug.imshow(np.hstack([bbsoi.draw_on_image(image, size=3), bbsoi.draw_on_image(image, size=3),
                                 bbsois_aug[0].draw_on_image(images_aug[0], size=3),
                                 bbsois_aug[1].draw_on_image(images_aug[1], size=3)]))
        pass

    pass


# C3： 不同类型的数据采用不同的增强器
class CopyingRandomStatesAndUsingMultipleAugmentationSequences(object):

    ##################################################################################################
    # https://nbviewer.jupyter.org/github/aleju/imgaug-doc/blob/master/notebooks/C03%20-%20Copying%20Random%20States%20and%20Using%20Multiple%20Augmentation%20Sequences.ipynb
    # Copying Random States and Using Multiple Augmentation Sequences
    ##################################################################################################

    @staticmethod
    def demo_heatmap():
        # load image + heatmap
        image = imgaug.quokka(size=0.2)  # uint8 array
        heatmap = imgaug.quokka_heatmap(size=0.2)  # HeatmapsOnImage object, contains a float array

        # show image + heatmap
        imgaug.imshow(np.hstack([image, heatmap.draw_on_image(image)[0]]))

        # print min/max of value ranges
        Tools.print("image min: %.2f, max: %.2f" % (np.min(image), np.max(image)))
        Tools.print("heatmap min: %.2f, max: %.2f" % (np.min(heatmap.get_arr()), np.max(heatmap.get_arr())))
        pass

    @staticmethod
    def demo_heatmap_problem():
        # load image + heatmap
        image = imgaug.quokka(size=0.2)  # uint8 array
        heatmap = imgaug.quokka_heatmap(size=0.2)  # HeatmapsOnImage object, contains a float array

        # our augmentation sequence: affine rotation, dropout, gaussian noise
        augs = imgaug_augmenters.Sequential([
            imgaug_augmenters.Affine(rotate=(-30, 30)),
            imgaug_augmenters.Dropout(0.03),
            imgaug_augmenters.AdditiveGaussianNoise(scale=10)
        ])

        # apply to image + heatmap
        augs_det = augs.to_deterministic()
        image_aug = augs_det.augment_image(image)
        heatmap_aug = augs_det.augment_image(heatmap.get_arr())

        # show results
        imgaug.imshow(np.hstack([image_aug, imgaug.HeatmapsOnImage(
            np.clip(heatmap_aug, 0.0, 1.0), shape=image_aug.shape).draw_on_image(image_aug)[0]]))

        # print min/max of value ranges
        print("image min: %.2f, max: %.2f" % (np.min(image_aug), np.max(image_aug)))
        print("heatmap min: %.2f, max: %.2f" % (np.min(heatmap_aug), np.max(heatmap_aug)))

        pass

    @staticmethod
    def demo_heatmap_manually_changing_parameter_values():
        # load image + heatmap
        image = imgaug.quokka(size=0.2)  # uint8 array
        heatmap = imgaug.quokka_heatmap(size=0.2)  # HeatmapsOnImage object, contains a float array

        # our augmentation sequence: affine rotation, dropout, gaussian noise
        augs = imgaug_augmenters.Sequential([
            imgaug_augmenters.Affine(rotate=(-30, 30)),
            imgaug_augmenters.Dropout(0.03),
            imgaug_augmenters.AdditiveGaussianNoise(scale=10)
        ])

        augs = augs.localize_random_state(recursive=True)
        augs_heatmap = augs.deepcopy()
        augs_heatmap[2].value = imgaug_parameters.Multiply(augs_heatmap[2].value, 0.001)

        image_aug = augs.augment_image(image)
        heatmap_aug = augs_heatmap.augment_image(heatmap.get_arr())

        imgaug.imshow(np.hstack([image_aug, imgaug.HeatmapsOnImage(
            np.clip(heatmap_aug, 0.0, 1.0), shape=image_aug.shape).draw_on_image(image_aug)[0]]))

        print("image min: %.2f, max: %.2f" % (np.min(image_aug), np.max(image_aug)))
        print("heatmap min: %.2f, max: %.2f" % (np.min(heatmap_aug), np.max(heatmap_aug)))
        pass

    @staticmethod
    def demo_heatmap_copy_random_state():
        # load image + heatmap
        image = imgaug.quokka(size=0.2)  # uint8 array
        heatmap = imgaug.quokka_heatmap(size=0.2)  # HeatmapsOnImage object, contains a float array

        sequence_images = imgaug_augmenters.Sequential([
            imgaug_augmenters.Affine(rotate=(-45, 45), name="affine"),
            imgaug_augmenters.Dropout(0.2, name="dropout"),
            imgaug_augmenters.AdditiveGaussianNoise(scale=20, name="gauss-noise")
        ])

        sequence_heatmaps = imgaug_augmenters.Sequential([
            imgaug_augmenters.Affine(rotate=(-45, 45), name="affine"),
            imgaug_augmenters.Dropout(0.2, name="dropout"),
            imgaug_augmenters.AdditiveGaussianNoise(scale=0.02, name="gauss-noise")  # different!
        ])

        sequence_images = sequence_images.localize_random_state(recursive=True)
        sequence_heatmaps_det = sequence_heatmaps.copy_random_state(sequence_images, matching="name")

        image_aug = sequence_images.augment_image(image)
        heatmap_aug = sequence_heatmaps.augment_image(heatmap.get_arr())
        print("image min: %.2f, max: %.2f" % (np.min(image_aug), np.max(image_aug)))
        print("heatmap min: %.2f, max: %.2f" % (np.min(heatmap_aug), np.max(heatmap_aug)))

        imgaug.imshow(np.hstack([image_aug, imgaug.HeatmapsOnImage(
            np.clip(heatmap_aug, 0.0, 1.0), shape=image_aug.shape).draw_on_image(image_aug)[0]]))
        pass

    @staticmethod
    def demo_heatmap_seed():
        # load image + heatmap
        image = imgaug.quokka(size=0.2)  # uint8 array
        heatmap = imgaug.quokka_heatmap(size=0.2)  # HeatmapsOnImage object, contains a float array

        imgaug.seed(1)  # to make Snowflakes reproducible

        sequence_images = imgaug_augmenters.Sequential([
            imgaug_augmenters.Affine(rotate=(-45, 45), random_state=1),
            imgaug_augmenters.Snowflakes(),  # added!
            imgaug_augmenters.Dropout(0.2, random_state=2),
            imgaug_augmenters.AdditiveGaussianNoise(scale=20, random_state=3)
        ], random_state=4)

        sequence_heatmaps = imgaug_augmenters.Sequential([
            imgaug_augmenters.Affine(rotate=(-45, 45), random_state=1),
            imgaug_augmenters.Dropout(0.2, random_state=2),
            imgaug_augmenters.CoarseDropout(0.2, size_px=4, random_state=100),  # added!
            imgaug_augmenters.AdditiveGaussianNoise(scale=0.02, random_state=3)
        ], random_state=4)

        image_aug = sequence_images.augment_image(image)
        heatmap_aug = sequence_heatmaps.augment_image(heatmap.get_arr())
        Tools.print("image min: %.2f, max: %.2f" % (np.min(image_aug), np.max(image_aug)))
        Tools.print("heatmap min: %.2f, max: %.2f" % (np.min(heatmap_aug), np.max(heatmap_aug)))

        imgaug.imshow(np.hstack([image_aug, imgaug.HeatmapsOnImage(
            np.clip(heatmap_aug, 0.0, 1.0), shape=image_aug.shape).draw_on_image(image_aug)[0]]))
        pass

    pass


if __name__ == '__main__':
    imgaug.seed(4)

    # 1
    # LoadAndAugmentAnImage.demo_2_augmenters()
    # LoadAndAugmentAnImage.demo_3_sequential()
    # LoadAndAugmentAnImage.demo_4_sequential_random_order()
    # LoadAndAugmentAnImage.demo_5_sequential_random_order_different_sizes()

    # 2
    # StochasticAndDeterministicAugmentation.demo_6_stochastic_and_deterministic_mode()
    # StochasticAndDeterministicAugmentation.demo_7_stochastic_and_deterministic_mode_keypoints()

    # 3
    # AugmentationOnMultipleCPUCores.demo_8_multicore()
    # AugmentationOnMultipleCPUCores.demo_9_multicore_non_image_data_keypoints()
    # AugmentationOnMultipleCPUCores.demo_10_multicore_pool()
    # AugmentationOnMultipleCPUCores.demo_11_multicore_using_pool_with_generators()
    # AugmentationOnMultipleCPUCores.demo_12_multicore_using_pool_with_generators_output_buffer_size()

    # 4
    # AugmentKeypointsLandmarks.demo_13_keypoint()
    # AugmentKeypointsLandmarks.demo_14_keypoint_aug()
    # AugmentKeypointsLandmarks.demo_15_keypoint_larger()
    # AugmentKeypointsLandmarks.demo_16_keypoint_shift()
    # AugmentKeypointsLandmarks.demo_17_keypoint_draw()

    # 5
    # AugmentBoundingBoxes.demo_18_bounding_box()
    # AugmentBoundingBoxes.demo_19_bounding_box_aug()
    # AugmentBoundingBoxes.demo_20_bounding_box_rotate()
    # AugmentBoundingBoxes.demo_21_bounding_box_rotate_highlight()
    # AugmentBoundingBoxes.demo_22_bounding_box_draw()
    # AugmentBoundingBoxes.demo_23_bounding_box_extract()
    # AugmentBoundingBoxes.demo_24_bounding_box_clip()
    # AugmentBoundingBoxes.demo_25_bounding_box_intersection_union_iou()
    # AugmentBoundingBoxes.demo_26_bounding_box_project()

    # 6
    # CopyingRandomStatesAndUsingMultipleAugmentationSequences.demo_heatmap()
    # CopyingRandomStatesAndUsingMultipleAugmentationSequences.demo_heatmap_problem()
    # CopyingRandomStatesAndUsingMultipleAugmentationSequences.demo_heatmap_manually_changing_parameter_values()
    # CopyingRandomStatesAndUsingMultipleAugmentationSequences.demo_heatmap_seed()

    # UsingImgaugWithMoreControlFlow.demo_non_deferred_way()
    # UsingImgaugWithMoreControlFlow.demo_single_function()
    # UsingImgaugWithMoreControlFlow.demo_single_function_and_different_input_types()

    # UsingProbabilityDistributionsAsParameters.demo_creating_and_using_stochastic_parameters()
    # UsingProbabilityDistributionsAsParameters.demo_coarse_salt_and_pepper()

    # AugmentPolygons.demo_polygons()
    # AugmentPolygons.demo_polygons_augmenter()
    # AugmentPolygons.demo_polygons_transforms_polygons_bounding_boxes()
    # AugmentPolygons.demo_many_consecutive_augmentations()
    # AugmentPolygons.demo_drawing_polygons()
    # AugmentPolygons.demo_extracting_image_content()
    # AugmentPolygons.demo_clipping_polygons()
    # AugmentPolygons.demo_computing_height_width_area()
    # AugmentPolygons.demo_modifying_polygon_start_point()
    # AugmentPolygons.demo_converting_to_bounding_boxes()
    # AugmentPolygons.demo_comparing_polygons()

    # AugmentHeatmaps.demo_loading_example_data()
    # AugmentHeatmaps.demo_augmenting_example_heatmap()
    # AugmentHeatmaps.demo_augment_heatmap_with_lower_resolution()
    # AugmentHeatmaps.demo_create_heatmap_from_numpy_array()
    # AugmentHeatmaps.demo_resize_average_pool_and_max_pool_heatmaps()
    # AugmentHeatmaps.demo_customize_drawing()

    # AugmentSegmentationMaps.demo_loading_example_data()
    # AugmentSegmentationMaps.demo_augment_example_segmentation_map()
    # AugmentSegmentationMaps.demo_scaling_segmentation_maps()
    # AugmentSegmentationMaps.demo_augment_segmentation_maps_smaller_than_their_corresponding_image()
    # AugmentSegmentationMaps.demo_draw_segmentation_maps()  # ???????????????????????
    # AugmentSegmentationMaps.demo_change_segmentation_maps_with_non_geometric_augmentations()
    pass
