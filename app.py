#!/usr/bin/env python3.6
from flask import Flask, request
from interpreter.interpreter import interpret_feedback, manual_testing

app = Flask(__name__)
app.config["DEBUG"] = False


@app.route('/rtf', methods=["GET", "POST"])  # what happens when some1 goes to /
def process():
    if request.method == "POST":
        feedback = str(request.form["feedback"])
        log_feedback(feedback)
        print("LOG: interpreting feedback:", feedback)
        result = interpret_feedback(feedback)
        result = {'warmer': 'wärmer gestellt', 'colder': 'kälter gestellt',
                  'neither': 'nicht verstellt'}[result]
        msg = f'Danke für dein Feedback! Die Heizung wurde {result}.'
        return '''<html>
                <body>
                    <p>Label = {result}</p>
                    <p><a href="/rtf">Neues Feedback geben</a>
                </body>
            </html>'''.format(result=msg)

    return '''
        <html>
          <head>
            <title> Deutsche Bahn - Feedback </title>
          </head>
            <body>
                <p>Stell dir vor, du sitzt in der Bahn und du frierst oder dir ist zu warm. Du entdeckst den QR-Code vor dir und kannst der bahn Feedback geben. 
<br/> <br/> 
Wie gefällt dir die Fahrt bei der Deutschen Bahn?:</p>
                <form method="post" action="./rtf">
                    <p><input name="feedback" /></p>
                    <p><input type="submit" value="Submit" /></p> 
                </form>
            </body>
        </html>
    '''


def log_feedback(feedback):
    with open('feedback.log', 'a') as file:
        file.write('\n' + feedback)


# Run locally for debugging
#if __name__ == "__main__":
#    app.run(host='0.0.0.0', port=5000, debug=True)
