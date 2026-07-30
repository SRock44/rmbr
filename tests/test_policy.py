from rmbr.policy import Policy


def test_strict_policy_allows_own_namespace_only():
    policy = Policy.strict()
    assert policy.can_read("coder", "coder") is True
    assert policy.can_read("coder", "researcher") is False
    assert policy.can_write("coder", "researcher") is False


def test_default_constructor_is_strict():
    policy = Policy()
    assert policy.can_read("coder", "researcher") is False


def test_open_policy_allows_everything():
    policy = Policy.open()
    assert policy.can_read("coder", "researcher") is True
    assert policy.can_write("coder", "researcher") is True


def test_allow_grants_specific_namespace_read():
    policy = Policy()
    policy.allow("coder", read="researcher")
    assert policy.can_read("coder", "researcher") is True
    assert policy.can_read("coder", "other") is False
    assert policy.can_write("coder", "researcher") is False  # read grant isn't a write grant


def test_allow_grants_wildcard_read():
    policy = Policy()
    policy.allow("supervisor", read="*")
    assert policy.can_read("supervisor", "anything") is True
    assert policy.can_read("supervisor", "literally-anything-else") is True


def test_allow_grants_list_of_namespaces():
    policy = Policy()
    policy.allow("coder", read=["researcher", "designer"])
    assert policy.can_read("coder", "researcher") is True
    assert policy.can_read("coder", "designer") is True
    assert policy.can_read("coder", "other") is False


def test_allow_write_grant():
    policy = Policy()
    policy.allow("supervisor", write="coder")
    assert policy.can_write("supervisor", "coder") is True
    assert policy.can_read("supervisor", "coder") is False  # write grant isn't a read grant


def test_allow_returns_self_for_chaining():
    policy = Policy()
    result = policy.allow("a", read="b").allow("c", read="d")
    assert result is policy
    assert policy.can_read("a", "b") is True
    assert policy.can_read("c", "d") is True


def test_on_access_can_override_default_deny():
    policy = Policy.strict()
    policy.on_access(lambda who, verb, ns, default: True)
    assert policy.can_read("coder", "researcher") is True


def test_on_access_can_override_default_allow():
    policy = Policy()
    policy.allow("coder", read="researcher")
    policy.on_access(lambda who, verb, ns, default: False)
    assert policy.can_read("coder", "researcher") is False


def test_on_access_returning_default_preserves_grant_behavior():
    policy = Policy()
    policy.allow("coder", read="researcher")
    calls = []

    def callback(who, verb, ns, default):
        calls.append((who, verb, ns, default))
        return default

    policy.on_access(callback)
    assert policy.can_read("coder", "researcher") is True
    assert policy.can_read("coder", "other") is False
    assert calls == [("coder", "read", "researcher", True), ("coder", "read", "other", False)]
