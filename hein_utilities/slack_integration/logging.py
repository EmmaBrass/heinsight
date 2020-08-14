from logging import StreamHandler, LogRecord
from .bots import SlackBot, RTMSlackBot


class SlackLoggingHandler(StreamHandler):
    def __init__(self,
                 # user_member_id: str,
                 bot_name: str,
                 token: str,
                 channel_name: str,
                 ):
        """
        A logging stream handler which logs to a slack channel. This class encompasses the RTMSlackBot class to enable
        real-time-messaging handling using the connected slack bot.

        :param str token: token to connect to slack client
        :param bot_name: name for the bot
        :param str channel_name: channel to message on. for example, #channelname
        """
        StreamHandler.__init__(self)
        self.bot = SlackBot(
            # user_member_id=user_member_id,
            bot_name=bot_name,
            token=token,
            channel_name=channel_name,
        )

    def emit(self, record: LogRecord) -> None:
        msg = self.format(record)
        self.bot.post_slack_message(msg)
        # todo catch and format different types of messages
        #   - see if there's a way to post files (catch and parse record?)


class RTMSlackLoggingHandler(StreamHandler):
    def __init__(self,
                 # user_member_id: str,
                 bot_name: str,
                 token: str,
                 channel_name: str,
                 ):
        """
        A logging stream handler which logs to a slack channel. This class encompasses the RTMSlackBot class to enable
        real-time-messaging handling using the connected slack bot.

        :param str token: token to connect to slack client
        :param bot_name: name for the bot
        :param str channel_name: channel to message on. for example, #channelname
        """
        StreamHandler.__init__(self)
        self.bot = RTMSlackBot(
            # user_member_id=user_member_id,
            bot_name=bot_name,
            token=token,
            channel_name=channel_name,
        )

    def emit(self, record: LogRecord) -> None:
        msg = self.format(record)
        self.bot.post_slack_message(msg)
        # todo catch and format different types of messages
        #   - see if there's a way to post files (catch and parse record?)

    def run_on(self, *, event: str):
        """a pass-through for the RTMClient.run_on decorator"""
        return self.bot.rtm_client.run_on(event=event)

    def start_rtm_thread(self):
        """starts the RTM monitor thread"""
        self.bot.start_rtm_client()
