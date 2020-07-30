'''
转到对数域，再进行傅里叶变换
'''

import logging
import numpy as np

# Homomorphic filter class
class HomomorphicFilter:
    """Homomorphic filter implemented with diferents filters and an option to an external filter.

    High-frequency filters implemented:
        butterworth
        gaussian

    Attributes:
        a, b: Floats used on emphasis filter:
            H = a + b*H


    """

    def __init__(self, a=0.5, b=1.5):
        '''

        :param a: rL
        :param b: rH - rL
        '''
        self.a = float(a)
        self.b = float(b)

    # Filters
    def __butterworth_filter(self, I_shape, filter_params):
        P = I_shape[0] / 2
        Q = I_shape[1] / 2
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U - P) ** 2 + (V - Q) ** 2)).astype(float)
        H = 1 / (1 + (Duv / filter_params[0] ** 2) ** filter_params[1])
        return (1 - H)

    def __gaussian_filter(self, I_shape, filter_params):
        P = I_shape[0] / 2
        Q = I_shape[1] / 2
        H = np.zeros(I_shape)
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U - P) ** 2 + (V - Q) ** 2)).astype(float)
        H = np.exp((-Duv / (2 * (filter_params[0]) ** 2)))
        return (1 - H)

    # Methods
    def __apply_filter(self, I, H):
        '''

        :param I: single channel image after log and fft
        :param H: one type of two HPFs
        :return: filtered image in frequency
        '''
        H = np.fft.fftshift(H)
        I_filtered = (self.a + self.b * H) * I
        return I_filtered

    def filter(self, I, filter_params, filter='butterworth', H=None):
        """
        Method to apply homormophic filter on an image

        Attributes:
            I: Single channel image
            filter_params: Parameters to be used on filters:

                butterworth:

                    filter_params[0]: Cutoff frequency
                    filter_params[1]: Order of filter

                gaussian:

                    filter_params[0]: Cutoff frequency

            filter: Choose of the filter, options:

                butterworth
                gaussian
                external

            H: Used to pass external filter

        """

        #  Validating image
        if len(I.shape) != 2:
            raise Exception('Improper image')
        # Take the image to log domain and then to frequency domain
        I_log = np.log1p(np.array(I, dtype="float"))
        I_fft = np.fft.fft2(I_log)

        # Filters
        if filter == 'butterworth':
            H = self.__butterworth_filter(I_shape=I_fft.shape, filter_params=filter_params)
        elif filter == 'gaussian':
            H = self.__gaussian_filter(I_shape=I_fft.shape, filter_params=filter_params)
        elif filter == 'external':
            print('external')
            if len(H.shape) != 2:
                raise Exception('Invalid external filter')
        else:
            raise Exception('Selected filter not implemented')

        # Apply filter on frequency domain then take the image back to spatial domain
        I_fft_filt = self.__apply_filter(I=I_fft, H=H)
        I_filt = np.fft.ifft2(I_fft_filt)
        I = np.exp(np.real(I_filt)) - 1
        return np.uint8(I)

# End of class HomomorphicFilter


if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt

    # Code parameters
    path_in = 'G:/2020KeTi/basic-learning/'
    path_out = 'C:/Users/lbw/Desktop/'
    img_path = 'frogMountain2.jpg'
    cv2.namedWindow('original image',cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('filtered image',cv2.WINDOW_AUTOSIZE)

    # Derived code parameters
    img_path_in = path_in + img_path
    img_path_out = path_out + 'filtered.png'

    # Main code
    img = cv2.imread(img_path_in)
    b,g,r = cv2.split(img)
    # for n in range(img.shape[2]):
    homo_filter = HomomorphicFilter(a=0.75, b=1.25)
    img_filtered_b = homo_filter.filter(I=b, filter_params=[30, 2])
    img_filtered_g = homo_filter.filter(I=g, filter_params=[30, 2])
    img_filtered_r = homo_filter.filter(I=r, filter_params=[30, 2])
    img_filtered = cv2.merge([img_filtered_b,img_filtered_g,img_filtered_r])

    img_filtered_gamma = np.power(img_filtered/255,0.6)
    cv2.imshow('original image',img)
    cv2.imshow('filtered image',img_filtered)
    cv2.imshow('filtered image gamma', img_filtered_gamma)
    # cv2.imwrite(img_path_out, img_filtered)
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()