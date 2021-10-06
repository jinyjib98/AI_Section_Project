from flask import Flask

def create_app():
    app = Flask(__name__)

    @app.route('/')
    def index():
        return 'Hello', 200
    
    @app.route('/morning')
    def hello():
        return 'good morning', 200
    
    return app



