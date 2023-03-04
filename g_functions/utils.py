from g_functions.postprocessors.knn_postprocessor import KNNPostProcessor
from g_functions.postprocessors.mcdropout_postprocessor import MCDropoutPostProcessor
from g_functions.postprocessors.mds_postprocessor import MDSPostProcessor
from g_functions.postprocessors.odin_postprocessor import ODINPostProcessor
from g_functions.postprocessors.base_postprocessor import BasePostProcessor


# Self-Ensemble
# conformal prediction

def get_postprocessor(postprocessor_name):
    postprocessors = {
        'knn': KNNPostProcessor,
        'mcdropout': MCDropoutPostProcessor,
        'mds': MDSPostProcessor,
        'odin': ODINPostProcessor,
        'msp': BasePostProcessor,
    }

    return postprocessors[postprocessor_name]