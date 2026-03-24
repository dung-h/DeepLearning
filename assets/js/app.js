(function () {
  const data = window.SITE_DATA;
  const body = document.body;

  if (!data || !body) {
    return;
  }

  const root = body.dataset.root || ".";

  function createElement(tag, className, text) {
    const element = document.createElement(tag);
    if (className) {
      element.className = className;
    }
    if (typeof text === "string") {
      element.textContent = text;
    }
    return element;
  }

  function renderMetaList(target, entries) {
    if (!target) {
      return;
    }

    target.innerHTML = "";
    entries.forEach((entry) => {
      const wrapper = document.createElement("div");
      wrapper.append(createElement("dt", "", entry.label), createElement("dd", "", entry.value));
      target.appendChild(wrapper);
    });
  }

  function createResourceLink(resource, isPrimary) {
    const link = createElement(
      "a",
      isPrimary ? "button-link" : "button-link button-secondary",
      resource.label
    );
    link.href = resource.url;
    link.target = "_blank";
    link.rel = "noreferrer";
    return link;
  }

  function renderResourceGroups(target, groups) {
    target.innerHTML = "";

    groups.forEach((group, groupIndex) => {
      const groupCard = createElement(
        "article",
        group.featured ? "resource-group-card is-featured" : "resource-group-card"
      );
      const heading = createElement("h3", "", group.title);
      const copy = createElement("p", "resource-group-copy", group.copy || "");
      const itemList = createElement("div", "resource-group-list");

      (group.items || [])
        .filter((item) => item && item.url)
        .forEach((resource, itemIndex) => {
          const item = createElement("div", "resource-item");
          const itemCopy = createElement("div", "resource-item-copy");

          if (resource.type) {
            itemCopy.appendChild(createElement("p", "resource-item-type", resource.type));
          }

          const itemTitle = createElement("h4", "", resource.title || resource.label);
          const itemNote = createElement("p", "", resource.note || "");
          itemCopy.append(itemTitle, itemNote);

          const actions = createElement("div", "resource-item-actions");
          actions.appendChild(
            createResourceLink(
              resource,
              Boolean(resource.featured || (group.featured && groupIndex === 0 && itemIndex === 0))
            )
          );

          item.append(itemCopy, actions);
          itemList.appendChild(item);
        });

      groupCard.append(heading, copy, itemList);
      target.appendChild(groupCard);
    });
  }

  function renderHome() {
    document.getElementById("site-title").textContent = `${data.course.code} | ${data.course.name}`;
    document.getElementById("site-subtitle").textContent = `${data.course.university} • ${data.course.faculty}`;
    document.getElementById("course-heading").textContent = `${data.course.code} · ${data.course.name}`;
    document.getElementById("course-summary").textContent = data.course.summary;
    document.getElementById("group-name").textContent = data.group.name;
    document.getElementById("group-summary").textContent = data.group.summary;
    document.getElementById("instructor-name").textContent = data.course.instructor;
    document.getElementById("home-overview-title").textContent = "Tổng quan";
    document.getElementById("overview-title").textContent = data.course.overview.title;
    document.getElementById("overview-copy").textContent = data.course.overview.copy;

    renderMetaList(document.getElementById("course-meta"), [
      { label: "Trường", value: data.course.university },
      { label: "Khoa", value: data.course.faculty },
      { label: "Giảng viên", value: data.course.instructor },
      { label: "Học kỳ", value: data.course.term },
    ]);

    const overviewPoints = document.getElementById("overview-points");
    overviewPoints.innerHTML = "";
    data.course.overview.points.forEach((point) => {
      overviewPoints.appendChild(createElement("li", "", point));
    });

    const membersGrid = document.getElementById("members-grid");
    membersGrid.innerHTML = "";
    if (data.group.members.length === 0) {
      membersGrid.appendChild(
        createElement(
          "p",
          "empty-state",
          "Danh sách thành viên sẽ được cập nhật khi nhóm hoàn thiện thông tin công khai."
        )
      );
    } else {
      data.group.members.forEach((member) => {
        const card = createElement("article", "member-card");
        card.append(createElement("h3", "", member.name), createElement("p", "member-role", member.role));
        membersGrid.appendChild(card);
      });
    }

    const assignmentGrid = document.getElementById("assignment-grid");
    assignmentGrid.innerHTML = "";
    data.assignments.forEach((assignment) => {
      const card = createElement("article", "assignment-card");
      const textWrap = createElement("div", "card-topline");
      const headingBlock = document.createElement("div");
      headingBlock.append(
        createElement("p", "section-kicker", assignment.numberLabel),
        createElement("h3", "", assignment.title)
      );
      textWrap.appendChild(headingBlock);

      const summary = createElement("p", "", assignment.cardSummary);
      const actions = createElement("div", "card-actions");
      const detailLink = createElement("a", "button-link", "Xem chi tiết");
      detailLink.href = `${root}/${assignment.page}`;
      actions.appendChild(detailLink);

      card.append(textWrap, summary, actions);
      assignmentGrid.appendChild(card);
    });
  }

  function renderAssignment() {
    const assignment = data.assignments.find((item) => item.id === body.dataset.assignmentId);
    if (!assignment) {
      return;
    }

    document.title = `${assignment.numberLabel} | ${data.course.code}`;
    document.getElementById("assignment-title").textContent = assignment.numberLabel;
    document.getElementById("assignment-name").textContent = assignment.title;
    document.getElementById("assignment-overview").textContent = assignment.overview;
    document.getElementById("assignment-summary").textContent = assignment.summary;

    renderMetaList(document.getElementById("assignment-meta"), [
      { label: "Tên bài", value: assignment.numberLabel },
      {
        label: "Thành viên",
        value: data.group.members.length
          ? data.group.members.map((member) => member.name).join(", ")
          : "Thông tin thành viên sẽ được cập nhật",
      },
      { label: "Giảng viên", value: data.course.instructor },
      { label: "Môn học", value: `${data.course.code} · ${data.course.name}` },
    ]);

    const resourceGrid = document.getElementById("resource-grid");
    const resourceEmpty = document.getElementById("resource-empty");
    const resources = Array.isArray(assignment.resources)
      ? assignment.resources.filter((item) => item.url)
      : [];
    const resourceGroups = Array.isArray(assignment.resourceGroups)
      ? assignment.resourceGroups.filter(
          (group) => Array.isArray(group.items) && group.items.some((item) => item.url)
        )
      : [];

    resourceGrid.innerHTML = "";
    if (resourceGroups.length > 0) {
      resourceEmpty.classList.add("hidden");
      renderResourceGroups(resourceGrid, resourceGroups);
    } else if (resources.length === 0) {
      resourceEmpty.classList.remove("hidden");
    } else {
      resourceEmpty.classList.add("hidden");
      resources.forEach((resource, index) => {
        const card = createElement("article", "resource-card");
        const actions = createElement("div", "resource-actions");
        actions.appendChild(createResourceLink(resource, index === 0));
        card.append(
          createElement("h3", "", resource.title || resource.label),
          createElement("p", "", resource.note || ""),
          actions
        );
        resourceGrid.appendChild(card);
      });
    }

    const sectionsPanel = document.getElementById("sections-panel");
    const sectionsGrid = document.getElementById("sections-grid");
    const sections = Array.isArray(assignment.sections)
      ? assignment.sections.filter((item) => item.title && item.copy)
      : [];

    sectionsGrid.innerHTML = "";
    if (sections.length === 0) {
      sectionsPanel.classList.add("hidden");
    } else {
      sectionsPanel.classList.remove("hidden");
      sections.forEach((section) => {
        const card = createElement("article", "resource-card");
        card.append(createElement("h3", "", section.title), createElement("p", "", section.copy));
        sectionsGrid.appendChild(card);
      });
    }
  }

  if (body.dataset.page === "home") {
    renderHome();
  }

  if (body.dataset.page === "assignment") {
    renderAssignment();
  }
})();