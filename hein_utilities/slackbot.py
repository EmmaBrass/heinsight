import warnings
from .slack_integration.bots import SlackBot, RTMSlackBot, NotifyWhenComplete

warnings.warn(
    'Slack bot classes have been refactored to .slack_integration.bots, please update your imports',
    DeprecationWarning,
    stacklevel=2,
)


if __name__ == '__main__':
    # todo convert into a slack bot example
    import time

    print('Wrapping function')

    @NotifyWhenComplete
    def foo(n, a='values'):
        print('inside test function')
        print(f'arg {n}, kwarg {a}')
        for i in range(n):
            print(i)
            time.sleep(1)

    print('Calling wrapped function')
    foo(2, 'kwarg value')
