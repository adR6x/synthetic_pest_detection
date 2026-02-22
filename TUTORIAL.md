# Tutorial: How to Get stated with this project for Contributers

Please read Readme for folder structure instructions.

## Shortcuts

Git merge code:
```
git merge main -m 'incorporating latest from main'
```
Starting Singular shell:
```
singularity shell --nv --bind /work:/work,/cwork:/cwork /opt/apps/containers/oit/jupyter/courses-jupyter-cuda.sif
``` 

## Setting up GitHub and Duke Cluster

Course shell here refers to your directory /hpc/<net_id>/ in Duke cluster.

1. We will first connect github to your duke cluster file system.
   Go to https://dcc-ondemand-01.oit.duke.edu/pun/sys/dashboard and 
   login with your NetID.

2. On the top bar select `Files` -> `Home Directory`. Create new folder
   with name `571_probab_ml`. Then enter the directory. You should see
   a button `>_ Open in Terminal` button on bar in second row, select it.

3. Login on the new terminal tab. You are now in the couse shell.

4. In your course shell, issue the following command to generate an
   SSH key for your container:
   ```
   ssh-keygen -t ed25519 -C "My probablilistic ml Key"
   ```
   You may access the default file by pressing Enter/Return on your
   
   keyboard.  If it asks you for a passphrase, DO NOT TYPE ANYTHING
   --- just press Enter/Return; and when it asks you for the
   passphrase again, just press Enter/Return again.  An ASCII image
   will appear in the Terminal, disregard that.

5. Now, we need to view the SSH key you generated. Run this command in
   your course shell:
   ```
   cat ~/.ssh/id_ed25519.pub
   ```
   You should see an output that begins with `ssh-ed25519`, ends with
   `My probablilistic ML Key`, and contains lots of random
   characters in between.

6. Highlight and copy the entire public SSH key beginning with
   `ssh-ed25519` and ending with `My probablilistic ML Key`
   (inclusive). Go to your Github Account Settings. You will find this
   by clicking on you Profile Pic on top right corner. Select `Settings` -> `SSH and GPG keys` -> `New SSH key`.
   - Set the Title to be "My Probablistic ML Key", or your liking.
   - Paste the copied public SSH key to text box `key`
   Press "Add SSH key" and you should be brought to a screen that confirms
   that the key was added to GitHub.
   > Note: The entire key should be copied as one single line, i.e.,
     with no line break (other than the fact that the long line may
     automatically wrap around).  If for some reason you end up with
     an extra line break, just delete that (but no other character).

7. In your course shell, run the following commands to set up `git`
   (replace `your_netid` and `your_name` below with appropriate
   values):
   ```
   git config --global user.email "your_netid@duke.edu"
   git config --global user.name "your_name"
   ```

8. Now, we are ready to pull the latest data/code from the course for
   the first time:
   ```
   git clone git@github.com:Mirsaid-ai/Synthetic-Data-Generation-for-Pest-Detection.git synthetic_data_gen_pest
   ```
   If all goes well, you should now see some contents in `synthetic_data_gen_pest`, which you can verify
   using:
   ```
   ls synthetic_data_gen_pest
   ```
   If this doesn't show any contents, there might be something wrong with
   your SSH key.

10. You are all set.

## Working on the Project

Let's say that you want to add a file "test.py" that will print "Hi Awesome people!"
Or work on scripts just like you would do on your local VS code.

In order to accomplish this task, you will learn:

* How to use the source control tool `git` to create a workspace for
  you to work on this feature without stepping on your teammates'
  toes, and to integrate new feature into your team repo in the end;

* Run python codes on Singular Shell so that you dont have to install
  required packages all the time.

## Creating a Python file and modifying the content

1. Lets start Code Server on the cluster. This is VS code on web browser.
   - Go to DCC website. Select on `Interactive Apps` -> `Code Server`
   - Select appropriate options. Then Launch.
   - You might have to wait for some time.
   - Change Code Server working folder to `571_probab_ml/synthetic_data_gen_pest`.
     
2. First, you want to create a "branch" to work on this new feature.
   Your team repo has a `main` branch that holds the "definitive" version
   of your project.  Creating a separate branch allows you to work making
   your changes to code in a protected environment where you and your
   teammates can work on things independently without stepping on each
   other's toes.  Pick a name for your branch that's meaningful to your
   team, say `test_XXX` (replace `XXX` with your own name to avoid
   nameclash with your teammates), create it, switch to it, and tell the
   team repo about it:
   ```
   git branch test_XXX
   git checkout test_XXX
   git push --set-upstream origin test_XXX
   ```
   We will walk you through the steps of checking in code changes next.

3. From the folders tab, create a new file named "test.py".
3. Open the file and type:
   ```
   print("Hi Awesome people")
   ```
4. Save the file

This step is analogous to making edits in files as we move ahead in the project.

## Your first `git` commit

1. You've done enough work to warrant a commit, which saves your work
   so far.  Here are some rules of thumbs on what goes into a commit:
   * Try to make your changes in conceptually clean steps, and commit
     at the end of every step.  Don't let too many changes accumulate!
   * Each step should be "complete" in some sense.  Don't check in
     a change that depends on something else not in the commit yet.
   * Always test your code before committing.  Every commit should
     end with a working code base.  Imagine how frustrating it would
     be if you check out somebody's changes only to find out that they
     break everything!

2. The following command gives you a summary of what changes you have
   made:
   ```
   git status
   ```
   You should see that you are currently on the `test_XXX`
   branch, and you have "untracked" `test.py` file. You can add all
   modifications into a staging area for commit in one go:
   ```
   git add -u
   ```
   You should add untracked files one at time, making sure that each
   one is really needed (as opposed to some temporary/sensitive file
   that shouldn't be checked in).  For example:
   ```
   git add test.py
   ```
   You can use the `git status` again to see where things stand.  It
   will show which files you've already added to the staging area for
   commit, and what other changes remain.

3. To commit, issue the following command:
   ```
   git commit -m "adding python file name test.py"
   ```
   Here, use the message in quotes to describe your changes briefly.
   Congrats!  You made your first commit (at least in your local
   repo).

4. Next, you need to push the commit upstream:
   ```
   git push
   ```
   Now, your changes will be visible in your team's repo for others to
   see, though at this point they still remain in branch separately
   from the `main` --- we will discuss how to do a "merge request"
   later.

## Keeping up with the `main` branch

1. So far, our changes are fairly local, but our next steps may
   involve editing other files in none-trivial ways, and these files
   might have changed since you branched off from `main`, causing
   potential conflicts.  If you don't expect there to be major changes
   affecting your own branch, you could wait until you are ready to
   merge your branch into `main` and resolve any conflicts at that
   time.  But what if signficant changes have been already made on
   `main`?  It would be prudent to incorporate these changes into your
   own branch soon rather than later.  We will walk through this
   scenario next.

2. Before you start, make sure that your branch is itself "clean"; you
   have committed and pushed all changes.  When you are ready, here is
   the sequence of `git` steps needed to refresh your branch:
   ```
   # first, temporarily switch to the main branch to download the latest:
   git checkout main
   git pull
   # then, go back to veena_dummy and merge the lastet main into it:
   git checkout test_XXX
   git merge main -m 'incorporating latest from main'
   git push
   ```
   The `git merge` step may fail because there are conflicts between
   your changes and those made to `main` that `git` cannot resolve.
   In that case, `git` will ask you to first fix conflicts manaully
   and then commit.  When that happens, you won't be able to do `git
   push` yet.  Instead, you must edit the files to resolve the
   conflicts.  When you open up one such file, you will see sections
   with conflicts marked as follows:
   ```
   <<<<<<< ...
   ... changes you made ...
   =======
   ... changes coming from the merge ...
   >>>>>>> ...
   ```
   To resolve the conflicts, you may have to discard either your
   changes or someone else's or doing a mixture of the two.  You will
   also need to delete the lines with `<<<<<<<`, `=======`, and
   `>>>>>>>`.  Once you are done with a file, `git add` it for commit.
   At any point, you can run `git status` to see what changes have
   already been staged for commit and what files remain unmerged.
   After you are all done, commit, and finally push.

## Running python files
1. Lets make some more changes to the `test.py` file and run it.
2. Open the `test.py` and copy following codes and save it:
   ```
   import torch
   print("Hi Awesome people")
   ```
3. To run this in our terminal, we will need to run our Terminal on sigular image format. This helps us with common packages. Run following on the terminal:
   ```
   singularity shell --nv --bind /work:/work,/cwork:/cwork /opt/apps/containers/oit/jupyter/courses-jupyter-cuda.sif
   ```
   This should show "Apptainer>" on your terminal.
3. Make sure the Apptainer is on the same directory as the python file using `pwd`. Now type
   ```
   python test.py
   ```
   This should print `Hi Awesome people`.


## Running Jupyter notebook files
1. Download Jupyter extension from Extension tab on Code server / VS code. It is the one from `ms-toolsai`
2. Create a file `test.ipynb`.
3. On the Terminal running on singular image format, type:
   ```
   jupyter server
   ```
   You will see some links like `http://127.0.0.1:8888/?token=c5df7f2d3a...`. Copy the one that you have on Terminal output.
4. You will see `Kernal` on the right-top of opened `test.ipynb` file. Press it. You will be asked to choose existing kernel or type in the link. Choose the link and paste the one you copied earlier on the command pallet.
5. Now it should be connected and ready to work!
7. You are the greatest person ever to alive for reading this tutorial until this point. You deserve all the happiness life can offer!! 🥳

## Merging your Edits into `main`. I still have not figured this out please do not follow instructions bellow!!

1. Make sure that your branch is clean (you've committed and push all
   the changes).  The last step of this process to open a "merge"
   requeset to your teammates so your changes can be incorporated into
   `main`.

   **IMPORTANT:** This step is tricky for this tutorial because all
   your teammates are working on the exact same feature (which
   shouldn't happen in practice) so it wouldn't make sense for
   everybody to merge.  As a team, to complete this tutorial, discuss
   as a team what you'd like to do: you may elect one member to merge
   his or her branch into `main`, or none will merge at all (because
   your own project may not need this wishlist feature).  Depending on
   what your team decides to do, you may skip Step 2 below and just to
   go Step 3.

2. To open a merge request, visit `https://gitlab.oit.duke.edu/`, find
   your team's project repo, click on "Merge Requests" in the left
   navigation bar, and then hit the blue "New merge request" button on
   the top right.  Then:
   * For the source branch, select `veena_dummy`.
   * For the target branch, select `main`.
   Once the merge request is created, ping your teammates to take a
   look at it.  If they think it's acceptable, they just need to
   approve the merge request and you are done!

3. Now that you are done.  You should now delete your branch --- it's
   considered a good `git` practice to delete a feature branch once
   it's done, instead of reusing it for other purposes.  If the branch
   has been merged, run:
   ```
   git branch -d veena_dummy
   ```
   Or if it hasn't been merged, run:
   ```
   git branch -D veena_dummy
   ```
   Then, delete that from the team repo as well:
   ```
   git push --delete origin veena_dummy
   ```
   :fireworks: Congrats on surviving this long tutorial!  Now get
   started on your real project!
