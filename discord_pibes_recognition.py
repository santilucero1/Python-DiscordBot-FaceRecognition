import discord
from discord.ext import commands, tasks
from discord.utils import get
from facepibes_recognition import url_to_image



bot=commands.Bot(command_prefix = "/")

@bot.event
async def on_ready():
	await bot.change_presence(status=discord.Status.idle ,activity=discord.Game("Test face recognition")) 
	print("bot:ok")

@bot.command()
async def who(ctx):

    #Get the http url from the img when you sent to chat
    url = ctx.message.attachments[0].url
    #Get the values of label an confidence
    label, confidence = url_to_image(url)

    await ctx.send('This is {} with a confidence value of {}'.format(label, confidence))
    await ctx.send(file=discord.File('path to the recogniced img')) #Example C:\Opencv Test\garbagefolder\imgrecogniced.jpg

bot.run("Token")
