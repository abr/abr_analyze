'''
Function for sending emails, with the option to attach images
'''
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import smtplib
import traceback

def send_email(from_email=None, from_password=None, to_email=None,
               subject=None, body=None, file_location=None):
    """
    Send an email with the option of adding an attachment

    PARAMETERS
    ----------
    from_email: string
        email address to send from
    from_password: string
        password to email address that the email is being sent from
        NOTE: can create a new email in gmail just for sending data if you
        are worried about security of your personal email address
    to_email: string
        email address to send to
    subject: string
        email subject / heading
    body: string
        main text / body of email
    file_location: string
        path to pdf or image to attach
    """
    if file_location is None:
        print("No attachment")
    if from_email is None:
        print("ERROR: Must provide source email to send from")
    if to_email is None:
        print("ERROR: Must provide email to send to")
    if from_password is None:
        print("ERROR: Must provide password for source email address")
    if subject is None:
        subject = "abr_control:results"
    if body is None:
        body = "This is an automated message using the abr_control repo"

    msg = MIMEMultipart()

    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    part = MIMEBase('application', 'octet-stream')

    if file_location is not None:
        filename = file_location
        attachment = open(filename, "rb")
        part.set_payload((attachment).read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', "attachment; filename= %s" % filename)
        msg.attach(part)

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, from_password)
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()

    except:  # pylint: disable=W0702
        print('Error: unable to send email')
        print(traceback.format_exc())
