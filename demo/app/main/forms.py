from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField, SelectField,MultipleFileField
from wtforms.validators import DataRequired, Length, Email, Regexp, ValidationError
from flask_wtf.file import FileField, FileAllowed, FileRequired
from flask_pagedown.fields import PageDownField

class UploadForm(FlaskForm):
    file = FileField('file', validators=[
        FileRequired(),
        FileAllowed(['zip'], 'zip only!')
    ])
    submit = SubmitField('上传')

#
# class UploadForm2(FlaskForm):
#     text = StringField('Text')
#     submit = SubmitField('Submit')
