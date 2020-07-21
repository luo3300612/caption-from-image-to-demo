import os

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hard to guess string'  # what for ?
    SQLALCHEMY_COMMIT_ON_TEARDOWN = True
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    FLASKY_ADMIN = os.environ.get("FLASKY_ADMIN") or "john@example.com"

    UPLOAD_FOLDER = '/home/luoyunpeng/ct/demo/app/static/uploadfiles'


    @staticmethod
    def init_app(app):
        pass

class DevelopmentConfig(Config):
    DEBUT = True
    SQLALCHEMY_DATABASE_URI = 'mysql+mysqlconnector://root:12345@localhost:3306/dev'


class TestingConfig(Config):
    TESTING = True
    WTF_CSRF_ENABLED = False
    SQLALCHEMY_DATABASE_URI = 'mysql+mysqlconnector://root:12345@localhost:3306/test'


class ProductionConfig(Config):
    SQLALCHEMY_DATABASE_URI = 'mysql+mysqlconnector://root:12345@localhost:3306/dev'

    @classmethod
    def init_app(cls, app):
        Config.init_app(app)

        import logging
        from logging.handlers import SMTPHandler
        credentials = ('591486669@qq.com', os.environ.get("EMAIL_PASSWORD"))
        secure = None
        if getattr(cls, 'MAIL_USERNAME', None) is not None:
            credentials = (cls.MAIL_USERNAME, cls.MAIL_PASSWORD)
            if getattr(cls, 'MAIL_USE_TLS', None):
                secure = ()
        mail_handler = SMTPHandler(
            mailhost=('smtp.qq.com', 25),
            fromaddr='591486669@qq.com',
            toaddrs=['591486669@qq.com'],
            subject='App Error',
            credentials=credentials,
            secure=secure
        )
        mail_handler.setLevel(logging.ERROR)
        app.logger.addHandler(mail_handler)


config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,

    'default': DevelopmentConfig
}
