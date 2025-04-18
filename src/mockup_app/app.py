from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

@app.route('/')
def index():
    return redirect(url_for('client_overview'))

@app.route('/client-overview')
def client_overview():
    return render_template('client_overview.html')

@app.route('/portfolio-analysis')
def portfolio_analysis():
    return render_template('portfolio_analysis.html')

@app.route('/constraints-editor')
def constraints_editor():
    return render_template('constraints_editor.html')

@app.route('/recommendations')
def recommendations():
    return render_template('recommendations.html')

if __name__ == '__main__':
    app.run(debug=True)