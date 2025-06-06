
An experimental language teacher that uses a realtime audio interface teach you any language you want to learn.

* Chat in any language you want to learn, using any native language.  For example, you can learn Spanish as a native German speaker.
* Converse about topics that interest you.
* It will automatically record words that you struggle with into a vocabulary database.
* Get drilled on your vocabulary words using [spaced repetition](https://en.wikipedia.org/wiki/Spaced_repetition#:~:text=Spaced%20repetition%20is%20an%20evidence,exploit%20the%20psychological%20spacing%20effect.)
* Since it's completely audio based, enjoy the ability to multi-task!  

## Requirements

1. OSX
2. You need an OpenAI SDK API key
3. A pair of headphones.  It does not work at all unless you use headphones.

## Installation

### Clone repo

```
git clone ..
```

### Install OSX deps

```
brew install ffmpeg portaudio
```

Actually I'm not sure if these are still needed.  You can try without it and see if you get any errors.

### Install uv

`uv` is a python package manager like `pip`.  To install it:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

More info on the [uv site](https://docs.astral.sh/uv/getting-started/installation/)

### Create a new uv venv (recommended)

This isn't absolutely required, but recommended to isolate your dependencies.

```
uv venv
source .venv/bin/activate
``` 

to create new venv.

### Sync python dependencies via uv sync

```
uv sync
```

### Setup OpenAI API key

⚠️ This uses an insane amount of tokens and is **very expensive** to run! ⚠️ 

⚠️ Don't leave it running and walk away!  You will quickly drain your OpenAI account ⚠️ 

```
cp .env.template .env
```

Now open .env and add your OpenAI key.  Ignore the other placeholders, you shouldn't need them.


## Running VibeFluent

Put your 🎧 headphones on, otherwise it won't work.  Then run it:

```
python main.py
```

It will ask you a few onboarding questions via the CLI interface, and then you should hear it speaking and you can answer it via voice.

## Setting up multiple user profiles

If you want to learn multiple languages, or you want a friend to try it with a fresh vocabulary database, you can create a new user profile by running:

```
python main.py --new-user
```

## Running vocab drill practice

At any point when talking to vibefluent, say:

"I want to do vocab drills"

and it will switch into vocab drill mode and quiz you on any vocab words its collected from your earlier conversations.

## Viewing the vocab database

Open `vibefluent.db` in your favorite sqlite UI.

## Backstory

This was inspired by [vibelingo](https://github.com/tleyden/vibelingo), which was created at the [SundAI Hack Berlin](https://lu.ma/lhzjraav) in May 2025.
