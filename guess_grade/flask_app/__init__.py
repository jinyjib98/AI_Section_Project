import os
from flask import Flask, render_template, request
import numpy as np
import pickle

CSV_FILEPATH = os.path.join(os.getcwd(), __name__, 'student-mat.csv')

def create_app():
    app = Flask(__name__)
    model = pickle.load(open('flask_app/model/grade_model.pkl', 'rb'))

    @app.route('/')
    def home():
        return render_template('home.html')
    
    @app.route('/answer-me', methods=['GET', 'POST'])
    def survey():
        return render_template('survey.html')
    
    # 이부분에서 막혀있음... 넘어가긴 하는데 아무것도 안뜨는건 뭐지...
    @app.route('/tada', methods=['POST'])
    def guess_grade():
        failures = request.form['failure']
        M_edu = request.form['mom']
        higher = request.form['higher']
        age = request.form['age']
        F_edu = request.form['pap']
        go_out = request.form['goout']
        romantic = request.form['romantic']
        travel = request.form['travel']

        survey = np.array([[failures, M_edu, higher, age, F_edu, go_out, romantic, travel]])
        pred = model.predict(survey)
        output = pred[0]
        return render_template('result.html', data=pred)
    

    from flask_app.model import model_bp
    app.register_blueprint(model_bp)
    
    return app

if __name__ == "__main__":
    app = create_app()

    app.run(debug=True)



