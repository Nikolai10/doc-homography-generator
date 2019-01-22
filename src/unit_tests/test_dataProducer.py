from unittest import TestCase

from dataProducer import DataProducer


class TestProducer(TestCase):

    target_dims = (384, 256)

    input = '../../res/input/'
    output = '../../output/'
    backgrounds = '../../res/backgrounds/'

    dataProducer = DataProducer(input, output, backgrounds)

    #----------------------------------------------------------------------------------------------------
    #                                              Test Producer                                        #
    #----------------------------------------------------------------------------------------------------

    # rgb generation
    def test_master(self):
        # adjust params in dataConfig.py before running script
        self.dataProducer.augmentDataset_master(max=1000, normalize=False, grayscale=False)

    #----------------------------------------------------------------------------------------------------
    #                                Component tests -> test_dataGenerator                              #
    #----------------------------------------------------------------------------------------------------
