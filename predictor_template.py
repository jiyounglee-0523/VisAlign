"""Template predictor for a VisAlign leaderboard submission with your own model.

Copy this file, fill in setup() and predict(), then run:

    python make_leaderboard_submission.py custom \
        --predictor my_predictor.py --output my_submission.json

and upload the resulting JSON on the "Submit" tab of
https://huggingface.co/spaces/jiyounglee0523/leaderboard
"""

CLASSES = ['tiger', 'zebra', 'camel', 'giraffe', 'elephant', 'rhino',
           'gorilla', 'bear', 'kangaroo', 'human', 'abstain']

model = None


def setup():
    """Optional. Called once before prediction starts — load your model here."""
    global model
    # Example:
    # model = MyModel.load_from_checkpoint('path/to/ckpt')
    # model.eval().cuda()


def predict(image, file_name):
    """Return your model's distribution for one open-test-set image.

    Args:
        image: PIL.Image (RGB) from the HF dataset jiyounglee0523/VisAlign
        file_name: original file name of the image (str)

    Returns:
        A list/array of 11 non-negative numbers over CLASSES
        (10 animal classes + abstain). It does not need to sum to 1;
        it is renormalized automatically. Return None if your pipeline
        cannot produce a distribution for this image (scored as abstention).
    """
    # Example with a classifier + abstention score:
    # x = my_transform(image).unsqueeze(0).cuda()
    # probs = torch.softmax(model(x), dim=-1).squeeze(0)          # 10 class probs
    # abstain = my_abstention_score(x)                            # in [0, 1]
    # return torch.cat([probs.cpu() * (1 - abstain),
    #                   torch.tensor([abstain])]).tolist()
    raise NotImplementedError('Implement predict() with your own model')
