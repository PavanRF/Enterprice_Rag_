import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.auth import (AuthError, LoginLimiter, authenticate, hash_password,  # noqa: E402
                      load_credentials, set_password, verify_password)
from rag.rbac import load_access_config  # noqa: E402


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class PasswordTests(unittest.TestCase):
    def test_hash_verifies_correct_password_only(self):
        stored = hash_password("correct horse battery")
        self.assertTrue(verify_password("correct horse battery", stored))
        self.assertFalse(verify_password("wrong password!!", stored))

    def test_same_password_gets_different_salts(self):
        self.assertNotEqual(hash_password("same-password-1"), hash_password("same-password-1"))

    def test_plaintext_is_not_stored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "creds.json"
            set_password(path, "priya.sharma", "a-long-secret-pw")
            self.assertNotIn("a-long-secret-pw", path.read_text())
            self.assertIn("priya.sharma", load_credentials(path))

    def test_short_password_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(AuthError):
                set_password(Path(tmp) / "c.json", "priya.sharma", "short", min_length=10)


class LoginTests(unittest.TestCase):
    def setUp(self):
        _, self.users = load_access_config(ROOT / "config/access.yaml")
        self.creds = {"priya.sharma": hash_password("finance-pass-123")}
        self.clock = FakeClock()
        self.limiter = LoginLimiter(max_failures=3, lockout_seconds=60, clock=self.clock)

    def test_correct_login_returns_user(self):
        user = authenticate(self.users, self.creds, "priya.sharma", "finance-pass-123", self.limiter)
        self.assertEqual(user.user_id, "priya.sharma")

    def test_wrong_password_and_unknown_user_give_same_error(self):
        with self.assertRaises(AuthError) as wrong:
            authenticate(self.users, self.creds, "priya.sharma", "nope-nope-nope", self.limiter)
        with self.assertRaises(AuthError) as unknown:
            authenticate(self.users, self.creds, "ghost.user", "nope-nope-nope", self.limiter)
        self.assertEqual(str(wrong.exception), str(unknown.exception))

    def test_user_without_password_cannot_log_in(self):
        with self.assertRaises(AuthError):
            authenticate(self.users, self.creds, "james.okoro", "anything-at-all", self.limiter)

    def test_lockout_after_repeated_failures_then_recovers(self):
        for _ in range(3):
            with self.assertRaises(AuthError):
                authenticate(self.users, self.creds, "priya.sharma", "bad-password-x", self.limiter)
        # Even the correct password is refused while locked.
        with self.assertRaises(AuthError) as locked:
            authenticate(self.users, self.creds, "priya.sharma", "finance-pass-123", self.limiter)
        self.assertIn("Too many", str(locked.exception))
        self.clock.now = 61.0
        user = authenticate(self.users, self.creds, "priya.sharma", "finance-pass-123", self.limiter)
        self.assertEqual(user.user_id, "priya.sharma")


class GoogleIdentityTests(unittest.TestCase):
    def setUp(self):
        from rag.identity import AuthError as _AuthError, user_from_google
        self.AuthError = _AuthError
        self.user_from_google = user_from_google
        _, self.users = load_access_config(ROOT / "config/access.yaml")

    def test_listed_email_resolves_regardless_of_case(self):
        email = self.users.get("priya.sharma").email
        self.assertEqual(self.user_from_google(self.users, email.upper()).user_id, "priya.sharma")

    def test_valid_google_account_not_on_list_is_denied(self):
        with self.assertRaises(self.AuthError):
            self.user_from_google(self.users, "stranger@gmail.com")

    def test_missing_email_is_denied(self):
        for value in [None, "", "  "]:
            with self.assertRaises(self.AuthError):
                self.user_from_google(self.users, value)


if __name__ == "__main__":
    unittest.main()
