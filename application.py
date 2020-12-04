from flask import Flask,request,render_template
from predict import classify_user

app = Flask(__name__)

@app.route("/",methods=["GET","POST"])
def home():
    if request.method == 'POST':
        uid = request.form.get("userid")
        print(uid)
        name,mbti,personality = classify_user(uid)
        message = "User not found"
        return render_template('analysis.html',display=True,name=name,mbti = mbti,personality=personality)

    return render_template('analysis.html',display=False)


if __name__ == "__main__":
    app.run(host="0.0.0.0",port="8000",debug = True)
