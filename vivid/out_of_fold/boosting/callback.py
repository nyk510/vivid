from lightgbm.callback import _format_eval_result, CallbackEnv


def logging_evaluation(logger, period=1, show_stdv=True):
    """
    create logging callback function

    :param logger:
    :param period:
    :param show_stdv:
    :return:
        function
    """
    def _callback(env: CallbackEnv):
        if period > 0 and env.evaluation_result_list and (env.iteration + 1) % period == 0:
            try:
                result = '\t'.join([_format_eval_result(x, show_stdv) for x in env.evaluation_result_list])
            except ValueError:
                scores = [f'{key} {score}' for key, score in env.evaluation_result_list]
                result = '\t'.join(scores)
            logger.info('[%d]\t%s' % (env.iteration + 1, result))

    _callback.order = 10
    return _callback
