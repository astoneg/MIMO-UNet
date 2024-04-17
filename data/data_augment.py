import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class PairRandomCrop(transforms.RandomCrop):

    def __call__(self, image, label):

        if self.padding is not None:
            image = F.pad(image, self.padding, self.fill, self.padding_mode)
            label = F.pad(label, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and image.size[0] < self.size[1]:
            image = F.pad(image, (self.size[1] - image.size[0], 0), self.fill, self.padding_mode)
            label = F.pad(label, (self.size[1] - label.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and image.size[1] < self.size[0]:
            image = F.pad(image, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)
            label = F.pad(label, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(image, self.size)

        return F.crop(image, i, j, h, w), F.crop(label, i, j, h, w)


class PairCompose(transforms.Compose):
    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class PairRandomHorizontalFilp(transforms.RandomHorizontalFlip):
    def __call__(self, img, label):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img), F.hflip(label)
        return img, label


class PairToTensor(transforms.ToTensor):
    def __call__(self, pic, label):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(pic), F.to_tensor(label)

class PairPadding():
    def __call__(self, image, label):
        """ Pads images that have dimensions that are not divisble by 4 to ensure they are.
            Since MIMO-UNet downsamples the inputs, we need to ensure that the dimension allow for the downsampling.
            Args:
                img (PIL Image): Image to be padded.

            Returns:
                PIL Image: Image padded to ensure that dimensions are divisible by 4.
            """
        pixels_to_pad_col = 4 - (image.size[0]%4)
        print(pixels_to_pad_col)
        print(image.size)
        pixels_to_pad_row = 4 - (image.size[1]%4)

        if pixels_to_pad_col != 4:
            image = F.pad(image, (pixels_to_pad_col, 0, 0, 0), padding_mode="edge")
            label = F.pad(label, (pixels_to_pad_col, 0, 0, 0), padding_mode="edge")
            print("col")
            print(image.size)
        if pixels_to_pad_row != 4:
            image = F.pad(image, (0, pixels_to_pad_row, 0, 0), padding_mode="edge")
            label = F.pad(label, (0, pixels_to_pad_row, 0, 0), padding_mode="edge")
            print("row")
            print(image.size)
        print(image.size)
        return image, label


