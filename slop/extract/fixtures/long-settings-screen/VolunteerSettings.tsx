export function VolunteerSettings() {
  return (
    <main className="space-y-10">
      <h1 className="text-2xl font-bold">Volunteer settings</h1>
      <p className="text-muted-foreground">Changes save as you go.</p>
      <section className="space-y-4">
        <h2 className="text-lg font-semibold">Account</h2>
        <div className="flex items-start justify-between">
          <div>
            <label className="font-medium">Display name</label>
            <p className="text-sm text-muted-foreground">Shown on rota sheets.</p>
          </div>
          <input type="checkbox" />
        </div>
        <div className="flex items-start justify-between">
          <div>
            <label className="font-medium">Email</label>
            <p className="text-sm text-muted-foreground">Used for sign-in links.</p>
          </div>
          <input type="checkbox" />
        </div>
        <div className="flex items-start justify-between">
          <div>
            <label className="font-medium">Phone</label>
            <p className="text-sm text-muted-foreground">Only shift leads can see it.</p>
          </div>
          <input type="checkbox" />
        </div>
        <button className="text-sm underline">Reset section</button>
      </section>
      <section className="space-y-4">
        <h2 className="text-lg font-semibold">Notifications</h2>
        <div className="flex items-start justify-between">
          <div>
            <label className="font-medium">Shift reminders</label>
            <p className="text-sm text-muted-foreground">Sent the evening before.</p>
          </div>
          <input type="checkbox" />
        </div>
        <div className="flex items-start justify-between">
          <div>
            <label className="font-medium">Swap requests</label>
            <p className="text-sm text-muted-foreground">When someone asks to swap with you.</p>
          </div>
          <input type="checkbox" />
        </div>
        <div className="flex items-start justify-between">
          <div>
            <label className="font-medium">Weekly summary</label>
            <p className="text-sm text-muted-foreground">Every Sunday at six.</p>
          </div>
          <input type="checkbox" />
        </div>
        <button className="text-sm underline">Reset section</button>
      </section>
      <section className="space-y-4">
        <h2 className="text-lg font-semibold">Availability</h2>
        <div className="flex items-start justify-between">
          <div>
            <label className="font-medium">Weekdays</label>
            <p className="text-sm text-muted-foreground">Tick the evenings you can do.</p>
          </div>
          <input type="checkbox" />
        </div>
        <div className="flex items-start justify-between">
          <div>
            <label className="font-medium">Weekends</label>
            <p className="text-sm text-muted-foreground">Mornings and afternoons are separate.</p>
          </div>
          <input type="checkbox" />
        </div>
        <div className="flex items-start justify-between">
          <div>
            <label className="font-medium">Holidays</label>
            <p className="text-sm text-muted-foreground">Block out dates you are away.</p>
          </div>
          <input type="checkbox" />
        </div>
        <button className="text-sm underline">Reset section</button>
      </section>
      <section className="space-y-4">
        <h2 className="text-lg font-semibold">Kitchen</h2>
        <div className="flex items-start justify-between">
          <div>
            <label className="font-medium">Food hygiene level</label>
            <p className="text-sm text-muted-foreground">Level 2 lets you plate up.</p>
          </div>
          <input type="checkbox" />
        </div>
        <div className="flex items-start justify-between">
          <div>
            <label className="font-medium">Allergens briefing</label>
            <p className="text-sm text-muted-foreground">Renew every twelve months.</p>
          </div>
          <input type="checkbox" />
        </div>
        <div className="flex items-start justify-between">
          <div>
            <label className="font-medium">Knife skills</label>
            <p className="text-sm text-muted-foreground">Optional, but it speeds up prep.</p>
          </div>
          <input type="checkbox" />
        </div>
        <button className="text-sm underline">Reset section</button>
      </section>
      <section className="space-y-4">
        <h2 className="text-lg font-semibold">Driving</h2>
        <div className="flex items-start justify-between">
          <div>
            <label className="font-medium">Van licence</label>
            <p className="text-sm text-muted-foreground">Needed for collections.</p>
          </div>
          <input type="checkbox" />
        </div>
        <div className="flex items-start justify-between">
          <div>
            <label className="font-medium">Mileage claims</label>
            <p className="text-sm text-muted-foreground">Paid monthly in arrears.</p>
          </div>
          <input type="checkbox" />
        </div>
        <div className="flex items-start justify-between">
          <div>
            <label className="font-medium">Parking permit</label>
            <p className="text-sm text-muted-foreground">Ask the office for a hanger.</p>
          </div>
          <input type="checkbox" />
        </div>
        <button className="text-sm underline">Reset section</button>
      </section>
      <section className="space-y-4">
        <h2 className="text-lg font-semibold">Privacy</h2>
        <div className="flex items-start justify-between">
          <div>
            <label className="font-medium">Profile photo</label>
            <p className="text-sm text-muted-foreground">Visible to other volunteers.</p>
          </div>
          <input type="checkbox" />
        </div>
        <div className="flex items-start justify-between">
          <div>
            <label className="font-medium">Hours history</label>
            <p className="text-sm text-muted-foreground">Kept for two years.</p>
          </div>
          <input type="checkbox" />
        </div>
        <div className="flex items-start justify-between">
          <div>
            <label className="font-medium">Data export</label>
            <p className="text-sm text-muted-foreground">A spreadsheet of your shifts.</p>
          </div>
          <input type="checkbox" />
        </div>
        <button className="text-sm underline">Reset section</button>
      </section>
      <section className="space-y-4">
        <h2 className="text-lg font-semibold">Training</h2>
        <div className="flex items-start justify-between">
          <div>
            <label className="font-medium">First aid</label>
            <p className="text-sm text-muted-foreground">Refresher every three years.</p>
          </div>
          <input type="checkbox" />
        </div>
        <div className="flex items-start justify-between">
          <div>
            <label className="font-medium">Manual handling</label>
            <p className="text-sm text-muted-foreground">Covers the big stock pots.</p>
          </div>
          <input type="checkbox" />
        </div>
        <div className="flex items-start justify-between">
          <div>
            <label className="font-medium">Till training</label>
            <p className="text-sm text-muted-foreground">For the Saturday café.</p>
          </div>
          <input type="checkbox" />
        </div>
        <button className="text-sm underline">Reset section</button>
      </section>
      <section className="space-y-4">
        <h2 className="text-lg font-semibold">Languages</h2>
        <div className="flex items-start justify-between">
          <div>
            <label className="font-medium">Spoken languages</label>
            <p className="text-sm text-muted-foreground">Helps us pair you with guests.</p>
          </div>
          <input type="checkbox" />
        </div>
        <div className="flex items-start justify-between">
          <div>
            <label className="font-medium">Sign language</label>
            <p className="text-sm text-muted-foreground">Any level is useful.</p>
          </div>
          <input type="checkbox" />
        </div>
        <button className="text-sm underline">Reset section</button>
      </section>
      <section className="space-y-4">
        <h2 className="text-lg font-semibold">Emergency contact</h2>
        <div className="flex items-start justify-between">
          <div>
            <label className="font-medium">Contact name</label>
            <p className="text-sm text-muted-foreground">Someone we can call on shift.</p>
          </div>
          <input type="checkbox" />
        </div>
        <div className="flex items-start justify-between">
          <div>
            <label className="font-medium">Contact phone</label>
            <p className="text-sm text-muted-foreground">Checked once a year.</p>
          </div>
          <input type="checkbox" />
        </div>
        <button className="text-sm underline">Reset section</button>
      </section>
      <section className="space-y-4">
        <h2 className="text-lg font-semibold">Danger zone</h2>
        <div className="flex items-start justify-between">
          <div>
            <label className="font-medium">Pause volunteering</label>
            <p className="text-sm text-muted-foreground">Hide yourself from rotas for a while.</p>
          </div>
          <input type="checkbox" />
        </div>
        <div className="flex items-start justify-between">
          <div>
            <label className="font-medium">Leave the kitchen</label>
            <p className="text-sm text-muted-foreground">Removes you from every rota.</p>
          </div>
          <input type="checkbox" />
        </div>
        <button className="text-sm underline">Reset section</button>
      </section>
    </main>
  );
}
