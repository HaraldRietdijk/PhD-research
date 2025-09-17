from examples_and_trials.models.tables_and_views import Stuff , MoreStuff , MyStuff
    
def test_view(app):
    app.session.add(Stuff(data="apples"))
    app.session.add(Stuff(data="pears"))
    app.session.add(Stuff(data="oranges"))
    app.session.add(Stuff(data="orange julius"))
    app.session.add(Stuff(data="apple jacks"))
    app.session.flush()
    app.session.commit()
    app.session.add(MoreStuff(stuff_id=3, data="foobar", waarde=2))
    app.session.add(MoreStuff(stuff_id=4, data="foobar", waarde=3))
    app.session.add(MoreStuff(stuff_id=3, data="foolbar", waarde=7))
    app.session.add(MoreStuff(stuff_id=4, data="foolbar", waarde=9))
    app.session.add(MoreStuff(stuff_id=1, data="foobar", waarde=5))
    app.session.add(MoreStuff(stuff_id=2, data="foobar", waarde=6))
    app.session.flush()
    app.session.commit()
    print(app.session.query(MyStuff).all())
