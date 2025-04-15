from flask import Flask, render_template, url_for, redirect, request, send_from_directory, flash
from nickmod import makeMessage, sayHello, makePlots, makeText
from forms import ContactForm

from exe_hello import run_hello

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


# @app.route('/signup', methods=['GET', 'POST'])
# def run_signup():
#     if request.method == 'POST':
#         name = request.form['name']
#         email = request.form['email']
#         body = request.form['body']

#         return render_template('confirm.twig', name=name, email=email, body=body)

#     form = ContactForm()
#     return render_template(
#         "contact.twig",
#         form=form,
#         template="form-template"
#     )

@app.route("/signup", methods=["GET", "POST"])
def run_signup():
    form = ContactForm()

    if request.method == 'POST':
        print("Form POSTed")
        print("Form data:", request.form)

    if form.validate_on_submit():
        print("Form validated successfully!")

        name = form.name.data
        email = form.email.data
        body = form.body.data

        download_filename = None
        if form.run_script.data:
            download_filename = run_hello()
            flash("Hello script executed!")

        return render_template(
            "confirm.twig",
            name=name,
            email=email,
            body=body,
            download_link=url_for('download_file', filename=download_filename) if download_filename else None
        )
    else:
        if request.method == 'POST':
            print("Form did NOT validate!")
            print("Form errors:", form.errors)

    return render_template("contact.twig", form=form, template="form-template")


@app.route('/download/<path:filename>')
def download_file(filename):
    return send_from_directory('.', filename, as_attachment=True)


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
