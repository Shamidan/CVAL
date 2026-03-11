"""
Познавательная дока для параметров
"""


class Description:
    retrain = 'perform a model retrain'
    limit = 'maximum number of records that will be returned'
    type_of_task = 'type of CV task. Currently supports: classification, detection, segmentation'

    time_start = 'start time of the operation in Unix timestamp'
    time_end = 'end time of the operation in Unix timestamp'

    dataset_name = 'the type of the dataset'
    dataset_description = 'the description of the dataset'
    dataset_id = 'id of the dataset'

    __pattern_action = 'whether to perform an action with '
    validation = __pattern_action + 'validation'
    training = __pattern_action + 'training'
    test = __pattern_action + 'test'
    classification = __pattern_action + 'classification'
    object_detection = __pattern_action + 'object_detection'
    expires_time = 'The Unix timestamp (in seconds) indicating the expiration time of the user account, ' \
                   'after which it will become inactive'
    is_active = 'Indicates whether the user is active or inactive'
    user_external_id = 'An external ID assigned to the user account for identification purposes'
    user_api_key = 'API User key for API access'
    use_val_test = 'True - the model is trained as long as the accuracy increases during validation.'
    weights_tag = 'The model weights_of_model ID used in the process. ' \
                  'Specify the "latest" tag to use the latest weights_of_model.'
    freeze_backbones = 'Whether to use backbone freezing in the training process'
    pretrain = 'Whether to use a pre-trained model or not'
    classification_backbone = 'type of the model. Currently supports: b0, resnet50, mobilenet'
    classification_method = 'selection strategy. Currently supports: least, margin, ratio, entropy, vae100, mixture'
    detection_backbone = 'type of the model. Currently supports: ...'
    detection_method = 'selection strategy. Currently supports: ...'
    detection_method_on_premise = 'selection strategy. Currently supports: margin, least, ratio, entropy'
    segmentation_backbone = 'Backbone of segmentation'
    segmentation_method = 'Active Learning method of object segmentation'
    normalized_error_value = 'Threshold for the normalized error _value'
    part_of_dataset = 'training, test, validation'

    # frames
    frame_ids_string = 'comma-separated list of image identifiers'

    # annotation
    type_of_annotation = 'classification, detection, segmentation'
    # result
    task_id = 'id of the task to perform polling'

    # weights_of_model id
    weights_id = 'weights Id to be used in active learning'
    update_weights = 'enables the option to retrain the existing model'
    weights_version = 'Weights Version to be used in the operation'
    # config params
    bbox_selection_policy = 'which bounding box to select when there are multiple boxes on an image, ' \
                            'according to their confidence. Currently supports: min, max, sum'
    bbox_selection_policy_on_premise = 'which bounding box to select when there are multiple boxes on an image, ' \
                                       'according to their confidence. Currently supports: min, max, sum, mean'
    bbox_selection_quantile_range = 'in what range of confidence will the bbox selection policy be applied'
    num_samples = 'absolute number of samples to select'
    batch_unlabeled = 'the limit of unlabeled samples that can be processed during selection'
    method = 'selection strategy. Currently supports ...'
    backbone = 'type of the model. Currently supports: ...'
    action = 'type of operation: sampling, test'
    weights = 'weights_of_model to be used in active learning or evaluation'
    labels_quantity = 'number of labels'

    old_version = 'previous version'
    new_weights_version = 'new version'
    frame_id = 'id of the frame'
    embedding_id = 'id of the embedding'
    embedding = 'frame embedding vector'
    probabilities = ('the probabilities for each object category are relative to a predicted bounding box.'
                     ' The order in the list is determined by the category number. sum must be = 1')
    prob_weights = ('Determines the significance (weight) of the prediction probability for each class. '
                    'The order in the list corresponds to the order of the classes.  It is essential for a multi-class entropy method.')

    access_key = 'Unique identifier used for authenticating access to your S3 storage.'
    secret_key = 'Confidential key used in conjunction with the Access Key to sign S3 requests for security.'
    scality_s3_endpoint = 'The URL where you can access your S3-compatible storage.'
    bucket_name = 'The unique name of the container where you store your objects.'
    storage_synchronize = 'Whether to perform automatic synchronization with cloud storage or not'
    n_epochs = 'number of model training epochs'
    use_validation = 'use_validation'
    early_stopping_num_epochs = 'number of epochs for early stopping. The number given here means that if the model does not improve within a given number of epochs, training should be aborted.'
    validation_step = 'number of epochs that must pass before validation. The number given here indicates how often validation needs to be performed'