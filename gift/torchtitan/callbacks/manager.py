from typing import Any, get_type_hints

from torchtitan.components.checkpoint import RESERVED_CHECKPOINT_KEYS
from torchtitan.protocols.sharding_backend import ShardingBackend
from torchtitan.callbacks.base import Callback, CallbackHandler, Event, MutableState


class CallbackManager:
    """Builds the callbacks a run configures, in the order it lists them.

    The composition root. A run's callbacks are a ``dict`` of configs: a
    preset adds a key to get a callback and leaves it out to not have one --
    there is no ``enable`` flag and nothing is built that then declines to
    run. Two entries of one class are two keys.
    """

    def __init__(
        self,
        callbacks: dict[str, Callback.Config],
        *,
        model_parts=None,
        sharding_backend: ShardingBackend,
        emits_step_events: bool = True,
    ) -> None:
        self.configs = callbacks
        self.handler = CallbackHandler(
            self._build(model_parts or [], sharding_backend=sharding_backend)
        )
        if not emits_step_events:
            offenders = self.handler.step_event_hooks()
            if offenders:
                raise ValueError(
                    "this trainer's train_step does not emit the step "
                    "callback events, but these enabled callbacks implement "
                    f"them and would silently never run: {', '.join(offenders)}. "
                    "Disable them, or emit the events from train_step."
                )

    def _build(
        self, model_parts, *, sharding_backend: ShardingBackend
    ) -> list[Callback]:
        runtime: dict[str, Any] = {
            "model_parts": model_parts,
            "sharding_backend": sharding_backend,
        }
        built: list[Callback] = []
        for name, config in self.configs.items():
            kwargs = {
                dep: (
                    runtime[dep]
                    if dep in runtime
                    else self._dependency(name, config, dep, built)
                )
                for dep in config.requires
            }
            built.append(config.build(**kwargs))
        return built

    @staticmethod
    def _dependency(
        name: str, config: Callback.Config, dep: str, built: list[Callback]
    ):
        """The already-built callback ``dep`` asks for, found by its type.

        By type and not by the key it was listed under: the key is the preset
        author's word, so resolving by it would break on a rename and would
        pick arbitrarily between two instances of the same callback. The type
        is the constructor's own annotation, so there is nothing to keep in
        sync.
        """
        try:
            owner = type(config).owner()
        except NotImplementedError:
            raise TypeError(
                f"callback '{name}': {type(config).__name__} carries no owner "
                "class; declare _owner_spec or bind the schema as the "
                "runtime callback's Config."
            ) from None
        # The owner's Config alias need not be a module global.
        hints = get_type_hints(owner.__init__, localns={"Config": owner.Config})
        want = hints.get(dep)
        if not (isinstance(want, type) and issubclass(want, Callback)):
            raise TypeError(
                f"{owner.__name__}.__init__ must annotate '{dep}' with the "
                f"callback type it needs; got {want!r}."
            )
        found = [callback for callback in built if isinstance(callback, want)]
        if not found:
            raise ValueError(
                f"callback '{name}' needs a {want.__name__}, which this run "
                "does not build before it. Add it to 'callbacks' above this "
                "entry: the order of the dict is the order they are built in."
            )
        if len(found) > 1:
            raise ValueError(
                f"callback '{name}' needs a {want.__name__} and this run "
                f"builds {len(found)}; which one it should read is not "
                "expressible. Keep one, or give it its own key to name."
            )
        return found[0]

    def find(self, want: type) -> Any | None:
        """The one built callback of type ``want``, or None.

        By type rather than by key: a callback's name in ``callbacks`` is the
        preset's choice and two presets are free to spell it differently.
        Refuses to guess when a run builds several -- the same rule
        ``_dependency`` applies when one callback asks for another.
        """
        found = [c for c in self.handler.callbacks if isinstance(c, want)]
        if not found:
            return None
        if len(found) > 1:
            raise ValueError(
                f"this run builds {len(found)} callbacks of type "
                f"{want.__name__}; which one to use is not expressible. Keep one."
            )
        return found[0]

    def emit(self, event: Event, trainer, **fields) -> MutableState:
        return self.handler.emit(event, trainer, **fields)

    def describe(self) -> str:
        return self.handler.describe()

    def step_event_hooks(self) -> list[str]:
        return self.handler.step_event_hooks()

    def close(self) -> None:
        self.handler.close()

    def stateful_objects(self) -> dict[str, object]:
        states = self.handler.stateful_objects()
        reserved = sorted(states.keys() & RESERVED_CHECKPOINT_KEYS)
        if reserved:
            raise ValueError(
                f"Callback checkpoint keys collide with the checkpointer's "
                f"own: {reserved}."
            )
        return states
