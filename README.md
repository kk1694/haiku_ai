# haiku_ai

Generating haikus with machine learning

## Samples

I generated 10 000 random samples in `samples.txt`. Note that the model was trained on haikus from reddit, so, some will contain foul language. A few highlights:

```

The United States 
 is not a state of affairs 
 in America . 

If you have the time 
 to see what the Bible says , 
 then God will exist . 

That 's why i 'm not sure 
 i have the right to express 
 my own opinion . 
 
And i 'd rather not 
 be able to work for you 
 than to be a dick . 
 
It 's a good movie . 
 My bad , i 'm so sorry you 
 had to endure it .
 
It 's a good reason 
 to be a cool person , but 
 you can always be . 
 
It 's a pretty good 
 idea to start a new 
 job for a while now . 
 
i would n't bother 
 with any of this if i 'm 
 not getting a job . 

```

The above represents the top 5%. There are many non-sensical ones for every decent haiku though. Interestingly, the shape is always correct (5-7-5 syllables).

## Generate New Samples

To generate new samples, just run

```
python generate_haikus.py 3
```

this prints out 3 haikus in the terminal. See all other options with `-h`.