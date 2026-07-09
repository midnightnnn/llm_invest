# Investment Chat Mobile UX Refinement

- Date: 2026-06-21
- Status: Approved for implementation

## Goal

Make the mobile investment chat feel like a reading-first chat surface after the user sends a message.

The current issues are:

- the input area stays too prominent after send
- long message bubbles look cluttered on small screens
- tapping and dragging inside message bubbles does not feel like normal transcript scrolling

## Intended behavior

1. After the user sends a message on mobile, the bottom input area collapses into a compact state.
2. The transcript gets more vertical room immediately after send.
3. Tapping the input area expands it again for the next message.
4. Vertical drag gestures on message bubbles remain available for scrolling.
5. Long message content wraps cleanly and keeps tables/code blocks readable on narrow screens.

## Scope

This change stays inside the ADK mobile override layer in `arena/ui/investment_chat_adk.py`.

No backend, routing, or database changes are needed.

## Implementation plan

- Add a mobile-only composer-collapsed body class.
- Toggle that class on send/submit, and clear it when the input is interacted with again.
- Hide the secondary action row in the collapsed state.
- Keep the main input usable in the collapsed state so a tap or new keystroke restores it.
- Reduce gesture suppression on message bubbles so touch scrolls are not blocked.
- Tighten mobile message wrapping and overflow handling.

## Non-goals

- No new backend state for composer mode.
- No new visible UI buttons for expanding or collapsing the composer.
- No redesign of the desktop chat surface.

## Verification

- Unit tests will check the injected CSS and JS strings.
- A local mobile smoke test should confirm:
  - send collapses the composer
  - tapping the input expands it again
  - message bubbles still scroll vertically
