from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from flask_wtf.recaptcha import RecaptchaField
from wtforms.validators import DataRequired, Length, Email

class ContactForm(FlaskForm):
    name = StringField(
        'Name',
        [DataRequired()]
    )
    email = StringField(
        'Email',
        [
            Email(message=('Not a valid email address.')),
            DataRequired()
        ]
    )
    body = StringField(
        'Message',
        [
            DataRequired(),
            Length(min=4,
            message=('Your message is too short.'))
        ]
    )
    recaptcha = RecaptchaField()
    submit = SubmitField('Submit')
