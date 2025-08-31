import momeutils 
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import mome_composition


app = Flask(__name__)
CORS(app)  # Allow requests from Obsidian

@app.route('/mOmEdRiAn', methods=['POST'])
def receive_canvas_path():


    try:
        data = request.get_json()
        canvas_path = data.get('canvas_path')
        selected_ids = data.get('selected_node_ids')
        
        if not canvas_path:
            return jsonify({'error': 'No canvas_path provided'}), 400
        if not selected_ids: 
            return jsonify({'error': 'Empty selection'}), 400
        
        # print(f"Received canvas path: {canvas_path}")*
        momeutils.crline('Executing graph for nodes: {}'.format(', '.join(selected_ids)))
        mome_composition.produce_composition(canvas_path, selected_ids)
        
        return jsonify({
            'message': f'Successfully received canvas path: {canvas_path}',
            'status': 'success'
        })
    
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Python server is running'})

if __name__ == '__main__':
    momeutils.crline("mOmEdRiAn ready...")
    app.run(host='0.0.0.0', port=5000, debug=True)
