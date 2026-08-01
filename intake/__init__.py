"""One door for everything entering the system, and one view of everything currently true.

Packaged rather than left in `cli/`, because the composition root was living in a dev tool:
`Session` was the only thing that assembled router + scene engine + car, so a real vehicle
integration would have had to reimplement wiring the CLI had already worked out.

Nothing is re-exported here on purpose. Import from the module that owns the shape
(`intake.envelope`, `intake.sources`, `intake.hub`), so that adding a consumer never widens
what importing `intake` pulls in — §12 names "intake becomes the god object" as the standing
risk of putting the composition root in one place.
"""
