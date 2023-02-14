########################################################################################################################
#   Proprietary and confidential.
#   Copyright (C) LuckMa, LLC - 2020, All Rights Reserved
#   Unauthorized access, copying or distributing any files, via any medium is strictly prohibited.
#
#   Last edited: 05/25/20
#
#   @author Jae Lim, <jae.lim@luckma.io>
########################################################################################################################

import os

from ml.code.util import misc_util, data_util


def calc_max_steps(ds_len, num_epochs, batch_size):
    num_examples_per_epoch = ds_len / num_epochs
    steps_per_epoch = num_examples_per_epoch / batch_size
    max_steps = int(num_epochs * steps_per_epoch)
    return max_steps


def get_model_dir(symbol):
    return data_util.maybe_create_dir(os.path.join(os.environ['MODEL_PATH'], symbol))


def get_model_export_dir(symbol):
    return data_util.maybe_create_dir(os.path.join(os.environ['MODEL_EXPORT_PATH'], symbol))


def get_valid_training_dates(dates, how='today'):
    if how == 'all':
        return dates
    elif how == 'today':
        return [misc_util.get_today_date()]
    elif misc_util.validate_date(how):
        return [how]  # only train on selected date
    else:
        raise ValueError(f'Unexpected training date: {how}')
