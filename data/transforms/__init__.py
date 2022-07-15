from enums import DATASETS
from .mnist import t_transforms as mnist_ttransforms
from .mnist import v_transforms as mnist_vtransforms

transforms={
    DATASETS.mnist: (mnist_ttransforms, mnist_vtransforms)
}
