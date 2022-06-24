from email.mime import text
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.encoders import encode_base64
import smtplib

def send_email(targets, email_body, date): 
    message = MIMEMultipart() # Initialize all the email details
    message["from"] = "My Company"
    message["to"] = ','.join([f'{target}' for target in targets])
    message["subject"] = f"Daily Feedback for TEMI Robot on {date}"
    message.attach(email_body)

    excel = MIMEBase('application', "octet-stream") # To send an Excel attachment
    with open("today_feedback.xlsx", "rb") as x:
        excel.set_payload(x.read())
    encode_base64(excel)
    excel.add_header('Content-Disposition', f'attachment; filename="{date}_feedback.xlsx"')
    message.attach(excel)

    with smtplib.SMTP(host = "smtp.gmail.com", port=587) as smtp:
        smtp.ehlo() # Hi SMTP server, Im here!
        smtp.starttls() # Start TLS (Encryption)
        smtp.login("email_in_gmail@gmail.com", "gmail_password") # Username, Password
        smtp.send_message(message)

def email_body(wall_text):
    mytext = MIMEText(wall_text) # This object takes the written text
    return mytext

if __name__=='__main__':
    from datetime import datetime

    text = email_body("""
    ---Write Text Here---
    """)

    send_email(["target1@gmail.com", "target2@hotmmail.com"], text, datetime.now())
