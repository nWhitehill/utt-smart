"""Bots that play Ultimate Tic Tac Toe, one package per bot.

Every bot package has a ``bot.py`` with a subclass of ``harness.Harness`` that
implements one method:

    move = bot.choose_move(game, to_move, deadline)

``game`` is a ``utt_smart.utt.UTT`` holding the current position, ``to_move`` is
'X' or 'O', and ``deadline`` is a ``time.monotonic()`` time the bot must answer
by. The move is a cell index in ``UTT``'s row-major numbering; the harness
converts it to the server's numbering. Anything slow to load goes in
``__init__``, which runs before the bot says ``ready``.

Each ``bot.py`` is also the bot's entrypoint on Bots Playing Games: run as a
script, it calls its class's ``main()``, which speaks the runner's protocol (see
``harness.py``). Packages only import their dependencies when they are used, so
a bot that needs torch doesn't stop the others running without it.
"""
import time


def search_until(search, deadline, step=16):
    # Runs search.start(step) until the deadline, so a bot uses the time it has
    # whatever the speed of the machine. Always runs at least one step, so the
    # root has children to pick from. Returns the number of iterations run
    iterations = 0
    while True:
        search.start(step)
        iterations += step
        if time.monotonic() >= deadline:
            return iterations
