"""
Lambda wrapper
"""

import json
from inference_onnx import ColaONNXPredictor

inferencing_instance = ColaONNXPredictor("./models/model.onnx")

def lambda_handler(event, context):
	"""
	Lambda function handler for predicting linguistic acceptability of the given sentence
	"""
	
	if "resource" not in event.keys():
		return inferencing_instance.predict(event["sentence"])
	body = event["body"]
	body = json.loads(body)
	print(f"Got the input: {body['sentence']}")
	response = inferencing_instance.predict(body["sentence"])
	return {
		"statusCode": 200,
		"headers": {},
		"body": json.dumps(response)
	}

if __name__ == "__main__":
	test = {"sentence": "this is a sample sentence"}
	lambda_handler(test, None)
