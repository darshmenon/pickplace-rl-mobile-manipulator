#!/usr/bin/env python3
"""
VLA Phase 3 - Language Parser Node
Upgraded: SmolLM2-360M-Instruct for flexible NLP with regex/keyword fallback.
"""

import re
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

# Fallback vocabulary
ACTIONS = ['pick', 'grab', 'grasp', 'place', 'put', 'move', 'transfer', 'sort', 'stack', 'clear']
COLORS  = ['red', 'blue', 'green', 'yellow', 'orange', 'white', 'black', 'purple', 'pink']
OBJECTS = ['cube', 'box', 'ball', 'cylinder', 'block', 'object', 'item', 'bottle', 'cup', 'mug']
PLACES  = ['tray', 'bin', 'basket', 'table', 'shelf', 'box', 'container', 'drop zone', 'left', 'right']

# Mutable container avoids global-reassignment false positives from static analysers
_LLM: dict = {'model': None, 'tokenizer': None, 'active': False}

_SYSTEM_PROMPT = (
    "You are a robotics instruction parser. "
    "Given a natural language instruction, output ONLY a valid JSON object with these exact keys: "
    "action (string), color (string or null), object (string), destination (string), confidence (float 0-1). "
    "No explanation. No markdown. Just the JSON object."
)


def _load_model() -> bool:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        model_id = 'HuggingFaceTB/SmolLM2-360M-Instruct'
        _LLM['tokenizer'] = AutoTokenizer.from_pretrained(model_id)
        _LLM['model'] = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map='cpu',
        )
        _LLM['model'].eval()
        _LLM['active'] = True
        return True
    except Exception:
        return False


def _parse_with_llm(text: str) -> dict | None:
    if not _LLM['active']:
        return None
    tokenizer = _LLM['tokenizer']
    model     = _LLM['model']
    try:
        import torch
        messages = [
            {'role': 'system', 'content': _SYSTEM_PROMPT},
            {'role': 'user',   'content': f'Instruction: {text}'},
        ]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors='pt')
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True
        ).strip()
        match = re.search(r'\{.*\}', generated, re.DOTALL)
        if match:
            result = json.loads(match.group(0))
            # Validate required keys are present; fall through to regex if not
            if all(k in result for k in ('action', 'object', 'destination')):
                result.setdefault('confidence', 0.85)
                return result
    except Exception:
        pass
    return None


def _parse_regex(text: str) -> dict:
    text_lower = text.lower()
    action = next((a for a in ACTIONS if a in text_lower), None)
    color  = next((c for c in COLORS  if c in text_lower), None)
    obj    = next((o for o in OBJECTS  if o in text_lower), None)
    place  = next((p for p in PLACES   if p in text_lower), None)

    transfer_match = re.search(
        r'(pick|grab|grasp)\s+(?:the\s+)?(\w+)\s+(?:and\s+)?(?:place|put|transfer)'
        r'\s+(?:it\s+)?(?:in|into|on|onto)\s+(?:the\s+)?(\w+)',
        text_lower
    )
    if transfer_match:
        action = 'pick_and_place'
        target_desc = transfer_match.group(2)
        destination = transfer_match.group(3)
        color = color or (target_desc if target_desc in COLORS else None)
        place = destination

    return {
        'action': action or 'unknown',
        'color': color,
        'object': obj or 'cube',
        'destination': place or 'tray',
        'confidence': 0.6,
        'parser': 'regex',
        'raw': text,
    }


def parse_instruction(text: str) -> dict:
    result = _parse_with_llm(text)
    if result:
        result['parser'] = 'smollm2'
        result['raw'] = text
        return result
    return _parse_regex(text)


class VLALanguageNode(Node):
    def __init__(self):
        super().__init__('vla_language_node')

        self.declare_parameter('use_llm', True)
        use_llm = self.get_parameter('use_llm').value

        if use_llm:
            self.get_logger().info('Loading SmolLM2-360M-Instruct (first run ~30s)...')
            if _load_model():
                self.get_logger().info('SmolLM2-360M-Instruct loaded. LLM parser active.')
            else:
                self.get_logger().warn(
                    'SmolLM2 unavailable (pip install transformers torch). Regex fallback active.'
                )
        else:
            self.get_logger().info('LLM disabled by parameter. Using regex parser.')

        self.srv = self.create_service(Trigger, '/vla/parse_instruction', self._parse_cb)
        self.text_sub = self.create_subscription(String, '/vla_instruction', self._instruction_cb, 10)
        self.cmd_pub  = self.create_publisher(String, '/vla/structured_command', 10)

        self.last_text = ''
        self.get_logger().info('VLA Language Node ready. Publish to /vla_instruction to start.')

    def _instruction_cb(self, msg: String):
        self.last_text = msg.data
        result = parse_instruction(msg.data)
        self.get_logger().info(f"[{result.get('parser', '?')}] → {result}")
        out = String()
        out.data = json.dumps(result)
        self.cmd_pub.publish(out)

    def _parse_cb(self, _, response):
        if self.last_text:
            result = parse_instruction(self.last_text)
            response.success = True
            response.message = json.dumps(result)
        else:
            response.success = False
            response.message = '{"error": "No instruction received yet"}'
        return response


def main(args=None):
    rclpy.init(args=args)
    node = VLALanguageNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
