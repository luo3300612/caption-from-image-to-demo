from datetime import datetime
import os
from flask import render_template, session, redirect, url_for, flash
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename

from . import main
from .forms import UploadForm
from flask import abort, request, current_app, make_response
import uuid
import glob

from ct.pipeline import gen_reports


@main.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    if form.validate_on_submit() and 'file' in request.files:
        folder_name = uuid.uuid4().hex
        save_path = os.path.join(current_app.config['UPLOAD_FOLDER'], folder_name)
        f = request.files.get('file')
        f.save(save_path + '.zip')
        output,img_paths = gen_reports(save_path + '.zip')
        img_paths = ['/'.join(img_path.split('/')[-4:]) for img_path in img_paths]
        return render_template('show.html', output=output, img_paths=img_paths)
    return render_template('index.html', form=form)
