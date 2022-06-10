import flask
import subprocess
import json
import os

def main(port_number=os.getenv('PORT')):
    app = flask.Flask(__name__)
    app.config['DEBUG'] = True

    @app.route('/', methods=['GET'])
    def start():
        return 'Flask begins here.'

    @app.route('/json', methods=['POST'])
    def json():
        config = flask.request.get_json()
        run_main(config)
        return 'Started'

    app.run(host='0.0.0.0', port=port_number)

if __name__== "__main__":
    main()
