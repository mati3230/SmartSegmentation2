import os
import json


class LookUpExpert:
    def __init__(self, path, key_method, action_method, look_up_old_key=False):
        self._look_up = {}
        self._path = path
        self._action_method = action_method
        self._key_method = key_method
        self._look_up_old_key = look_up_old_key
        self._load()

    def _load(self):
        if not os.path.isfile(self._path):
            return
        with open("./expert.json", "r") as f:
            self._look_up = json.load(f)

    def _save(self):
        jsn = json.dumps(self._look_up)
        f = open(self._path, "w")
        f.write(jsn)
        f.close()

    def _action(self, env, state):
        key = self._key_method(state)
        new_key = not (key in self._look_up)
        action = None
        if new_key:
            # print("new key", key)
            action = self._action_method(env, state)
            self._record(key, action)
        else:
            if self._look_up_old_key
                action = self._look_up[key]
            else:
                action = self._action_method(env, state)
        return action, key, new_key

    def _record(self, key, action):
        self._look_up[key] = action
        self._save()

    def __call__(self, env, state):
        return self._action(env, state)
