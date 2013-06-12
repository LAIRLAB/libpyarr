import os, sys, smtplib, signal, argparse

import common.util.file_util as fu
import common.util.misc_util as mu
import common.util.color_printer as cpm

from email.mime.text import MIMEText

default_recipients = []

class himmailer_superparser(object):
    @staticmethod
    def create_superparser(parent):
        p = argparse.ArgumentParser(parents = [parent], add_help = False)
        p.add_argument('-r', '--recipients', nargs = '+')
        p.add_argument('--email-message-id', default = '')
        return p
    
    @staticmethod
    def validate(args):
        return True

class HIMMailer(object):
    def __init__(self, sender = 'himsender2013@gmail.com', u = 'himsender2013', p = 'without3explosion'):
        self.sender = sender
        self.username = u
        self.password = p
        
    def send_email(self, recipients, subject = 'HIMMailer', text = ''):
        if recipients is None or len(recipients) == 0:
            cpm.gcp.warning("No recipients specified, not emailing")
            return
        if isinstance(recipients, str):
            recipients = [recipients]
        subject = '{}'.format(subject)
        try:
            server = smtplib.SMTP('smtp.gmail.com:587')
            server.starttls()

            msg = MIMEText(text.translate(None, '[]'))
            msg['Subject'] = subject
            msg['From'] = self.sender
            msg['To'] = ', '.join(recipients)
            server.login(self.username, self.password)
            server.sendmail(self.sender, recipients, msg.as_string())
        except:
            cpm.gcp.error("Could not send mail to: {}. Error: {}".format(recipients, sys.exc_info()[0]))
            
    def register_signal_handling(self, recipients, s):
        signal.signal(signal.SIGTERM, lambda signum, frame: self._eqp(recipients,
                                                                     subject = s))
                     
    def _eqp(self, recipients, subject):
        print "Caught signal. Sending email to: {}...".format(recipients)

        loc_string = 'waiting for user decision at: {}'.format(mu.get_user_at_host())
        self.send_email(recipients, 
                        subject ='{}, {}, {}'.format(subject, 
                                                     loc_string,
                                                     fu.gts()))
        if raw_input('Actually quit this job? (y / N): ').lower()[0] == 'y':
            print "Quitting"
            sys.exit()
        else:
            print "Ignoring signal, forging ahead"

def send_email(recipients, subject, body = ''):
    HIMMailer().send_email(recipients, subject, body)
