from torchvision import transforms


def get_transform(
    img_size: int, mean: float, std: float, grayscale: bool = True
) -> transforms.Compose:
    """
    This function returns a list of transformations
    to be applied to the images.

    parameters
    ----------
    img_size : int
        The size of the image

    mean : float
        The mean of the dataset

    std : float
        The standard deviation of the dataset

    grayscale : bool
        Whether to convert the image to grayscale
    """

    transform_list = []

    if grayscale:
        transform_list.append(transforms.Grayscale(num_output_channels=1))

    transform_list.extend(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean], std=[std]),
        ]
    )

    return transforms.Compose(transform_list)
