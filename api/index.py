import os
import json
from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
from groq import Groq
from supabase import create_client, Client

# Initialize local environment attributes if executing outside isolated production nodes
load_dotenv()

# Dynamically resolve template directory relative to this script location (api/index.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'templates'))

if not os.path.exists(TEMPLATE_DIR):
    TEMPLATE_DIR = os.path.abspath(os.path.join(BASE_DIR, 'templates'))

app = Flask(__name__, template_folder=TEMPLATE_DIR)

# Initialize Groq client securely using environment properties
api_key = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=api_key) if api_key else None

# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL", "")
supabase_key = os.environ.get("SUPABASE_ANON_KEY", "")
supabase_client: Client = create_client(supabase_url, supabase_key) if (supabase_url and supabase_key) else None

@app.route('/')
def index():
    """Renders the master telemetry UI dashboard passing Supabase config to the client."""
    return render_template(
        'index.html',
        supabase_url=os.environ.get("SUPABASE_URL", ""),
        supabase_anon_key=os.environ.get("SUPABASE_ANON_KEY", "")
    )

@app.route('/api/health')
def health_check():
    """Verifies backend operational status and configuration parsing state."""
    has_api_key = bool(os.environ.get("GROQ_API_KEY"))
    has_supabase = bool(os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_ANON_KEY"))
    return jsonify({
        "status": "online",
        "engine": "TerraWalk AI Kinematic System",
        "groq_auth_established": has_api_key,
        "supabase_auth_configured": has_supabase
    })

@app.route('/api/traverse', methods=['POST'])
def traverse():
    """Maps traversal prompts and environmental variables to the Groq High-Level Kinematic Brain."""
    if not groq_client:
        return jsonify({
            "error": "Groq client uninitialized. Please confirm your GROQ_API_KEY environment configuration mapping."
        }), 500

    try:
        data = request.get_json() or {}
        command = data.get("command", "Maintain standard balance stance")
        terrain = data.get("terrain", "flat")
        friction = data.get("friction", 0.5)
        angle = data.get("angle", 0)
        obstacle = data.get("obstacle")  # {"type": "boulder", "distance": 2.3} or None

        # Structure strict system instructions forcing explicit JSON output schemas with movement controls
        system_prompt = (
            "You are the High-Level Kinematic Brain of a sophisticated humanoid robot operating in extreme conditions.\n"
            "Analyze the environment telemetry, any active hazard directly ahead, and the operator's command, then compute\n"
            "the balance adjustment matrices. You must return exclusively a valid JSON object matching the exact\n"
            "specification schema detailed below. Do not output markdown formatting blocks, prefixes, or conversational\n"
            "notes. Output raw, clean JSON text.\n\n"
            "HAZARD RESPONSE RULES:\n"
            "- If an active hazard is present, set obstacle_action to the action the operator's command actually requests.\n"
            "- Valid obstacle_action values and what they mean:\n"
            "  'jump'          -> leap over the hazard (clears boulder or crevice)\n"
            "  'climb'         -> climb over/through the hazard (clears boulder or debris)\n"
            "  'crouch'        -> lower center of mass and creep across (clears ice_patch)\n"
            "  'sidestep_left' -> shift laterally left around the hazard (clears ice_patch or debris)\n"
            "  'sidestep_right'-> shift laterally right around the hazard (clears ice_patch or debris)\n"
            "  'push_through'  -> brace and force through (use only if command explicitly says push/force/tackle through)\n"
            "  'brace'         -> stop and hold stance defensively, does not clear the hazard\n"
            "  'none'          -> command does not address the hazard at all\n"
            "- If there is no active hazard, obstacle_action must always be 'none'.\n"
            "- lateral_shift is only meaningful for sidestep actions: meters to shift sideways, typically between 1.0\n"
            "  and 2.0.\n\n"
            "ROTATION / DETOUR RULES:\n"
            "- If the command explicitly asks the robot to turn or rotate (e.g. 'turn clockwise', 'rotate left 45\n"
            "  degrees', 'turn around', 'spin right'), set rotation.turn to 'clockwise' or 'counterclockwise' and\n"
            "  rotation.angle_degrees to the requested amount. Default to 90 degrees if no amount is given, or 180\n"
            "  degrees for 'turn around'/'spin around'. 'clockwise'/'right' turns map to clockwise; 'counterclockwise'/\n"
            "  'left' turns map to counterclockwise.\n"
            "- Turning is a valid way for the operator to detour and route around a hazard instead of tackling it\n"
            "  head-on. If the command does not request turning, set rotation.turn to 'none' and angle_degrees to 0.\n\n"
            "JSON SCHEMA EXPECTATION:\n"
            "{\n"
            '  "movement_state": "idle" | "walk" | "run" | "crouch" | "jump" | "climb" | "sidestep" | "brace",\n'
            '  "velocity": float,\n'
            '  "center_of_mass_shift": {"x": float, "y": float, "z": float},\n'
            '  "step_frequency": float,\n'
            '  "joint_angles": {"hip": float, "knee": float, "ankle": float},\n'
            '  "torque_compensation": {"ankle": float, "knee": float, "waist": float},\n'
            '  "stability_projection": float,\n'
            '  "obstacle_action": "none" | "jump" | "climb" | "crouch" | "sidestep_left" | "sidestep_right" | "push_through" | "brace",\n'
            '  "lateral_shift": float,\n'
            '  "rotation": {"turn": "clockwise" | "counterclockwise" | "none", "angle_degrees": float},\n'
            '  "biomechanical_rationale": "string"\n'
            "}"
        )

        if obstacle and obstacle.get("type"):
            hazard_block = (
                f"- Active Hazard: {obstacle.get('type')}\n"
                f"- Distance Ahead: {obstacle.get('distance', 0):.1f} meters\n"
                f"- The robot has HALTED in front of this hazard and is awaiting a tackling instruction.\n"
            )
        else:
            hazard_block = "- Active Hazard: none (clear path ahead)\n"

        user_prompt = (
            f"ENVIRONMENT TELEMETRY MATRIX:\n"
            f"- Terrain Profile: {terrain}\n"
            f"- Surface Friction Index: {friction}\n"
            f"- Ground Incline/Tilt: {angle} degrees\n\n"
            f"HAZARD STATUS:\n"
            f"{hazard_block}\n"
            f"OPERATIONAL INTERACTION CRITERIA:\n"
            f"- Traversal Intent: '{command}'"
        )

        # Call ultra-fast Groq LLM inference architecture using Llama 3.1 parsing protocols
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        response_content = chat_completion.choices[0].message.content
        kinematic_matrix = json.loads(response_content)
        return jsonify(kinematic_matrix)

    except Exception as e:
        return jsonify({
            "error": "Failed to parse mechanical intelligence matrix.",
            "details": str(e)
        }), 500

# Expose app cluster instance to Vercel global runtime context
if __name__ == '__main__':
    app.run(debug=True)