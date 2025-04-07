from flask import Flask, render_template, url_for, redirect, request
from nickmod import makeMessage, sayHello, makePlots, makeText
from forms import ContactForm


app = Flask(
    __name__,
    template_folder="templates",
)


app.secret_key = 'my secret key'
app._static_folder = 'static'


@app.route('/')
def run_index():
    date = sayHello()

    context = {
        'date': date,
        'age': '12'
    }

    return render_template("index.twig", **context)


@app.route('/report')
def run_report():
    text = makeText()
    return text


@app.route('/plots')
def run_plots():
    output = makePlots()

    context = {
        'output': output,
    }

    return render_template("plots.twig", **context)


@app.route('/message/<message>')
def run_message(message):
    message = makeMessage(message)
    return message


@app.route('/signup', methods=['GET', 'POST'])
def run_signup():
    if request.method == 'POST':
        name = request.form['name']
    return name


@app.route("/contact", methods=["GET", "POST"])
def contact():
    form = ContactForm()
    if form.validate_on_submit():
        return redirect(url_for("signup"))
    return render_template(
        "contact.twig",
        form=form,
        template="form-template"
    )


if __name__ == "__main__":
    app.run(debug=True)
