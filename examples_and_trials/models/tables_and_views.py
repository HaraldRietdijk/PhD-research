from sqlalchemy import Column, Index, VARCHAR, Integer, ForeignKey, select, func, and_
from database.models.model import DBActivityModel
from database.models.view import view

class Stuff(DBActivityModel):
    __tablename__ = "stuff"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    data = Column(VARCHAR(50))
    __table_args__ = (Index('INDEX1', "data"), Index('INDEX2', "data", "id"), )

class MoreStuff(DBActivityModel):
    __tablename__ = "more_stuff"
    
    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    stuff_id = Column(Integer, ForeignKey(Stuff.id))
    data = Column(VARCHAR(50))
    waarde = Column(Integer)

stuff = DBActivityModel.metadata.tables['stuff']
more_stuff = DBActivityModel.metadata.tables['more_stuff']
sum_stuff = view(
    "sum_stuff",
    DBActivityModel.metadata,
    select(
      func.sum(more_stuff.c.waarde).label("sum_waarde"), 
      more_stuff.c.stuff_id.label("stuff_id")
      )
    .select_from(more_stuff)
    .group_by(more_stuff.c.stuff_id)
    )

stuff_view = view(
    "stuff_view",
    DBActivityModel.metadata,
    select(
        stuff.c.id.label("id"),
        stuff.c.data.label("data"),
        more_stuff.c.data.label("moredata"),
        more_stuff.c.waarde.cast(VARCHAR).label("charwaarde"),
    )
    .select_from(stuff.join(more_stuff))
    .where(and_(11<sum_stuff.c.sum_waarde, more_stuff.c.stuff_id==sum_stuff.c.stuff_id))
    .order_by(stuff.c.data, more_stuff.c.data),
)

class MyStuff(DBActivityModel):
    __table__ = stuff_view

    __mapper_args__ = { 'primary_key':['moredata', 'id', 'data']}

    def __repr__(self):
        return f"MyStuff({self.id!r}, {self.data!r}, {self.moredata!r}, {self.charwaarde!r})"
